# S17 Teammate 生命周期核验与改进建议

## 结论

现场中的 `in_progress` 卡死不是模型单纯忘记调用 `complete_task`，而是 teammate 在自动认领成功后发生未捕获异常并退出。任务状态已经写成 `in_progress`，但线程没有机会继续工作或执行清理；之后 Lead 发出的消息只能堆积在 inbox。

## 现场证据

- `task_1783053732_9057`：12:42:21 被 alice 认领，保持 `in_progress`。
- `task_1783053732_4068`：12:42:22 被 bob 认领，保持 `in_progress`。
- alice/bob inbox 在 12:48 收到多条继续工作和 complete_task 指令，但一直未消费。
- 检查时没有存活的 Python teammate 进程。
- 原实现的 `scan_unclaimed_teammates()` 返回 task ID 字符串；`idle_poll()` 随后访问 `task['id']` 和 `task['subject']`，触发 `TypeError`。
- 异常发生在 claim 写盘之后，因此留下“有 owner、无 worker”的 orphan task。
- teammate run 函数没有覆盖所有出口的 `finally`，异常时不会从 `active_teammates` 清理，也不会释放任务。
- S16 遗留的无限 inbox idle loop 位于 S17 `idle_poll()` 之前，正常情况下 S17 的任务板轮询无法执行。

## 推荐生命周期

```text
SPAWN
  -> WORK (最多 10 个 LLM/tool turn)
  -> IDLE (inbox 优先，然后扫描任务板)
       -> WORK       收到普通消息或成功认领任务
       -> SHUTDOWN   收到 shutdown_request
       -> TIMEOUT    空闲超时
  -> FINALIZE
       -> best-effort summary
       -> 释放仍为 in_progress 的 owned tasks
       -> 从 active registry 移除
       -> 写 thread_stopped 日志
```

所有退出路径，包括 API 错误、工具异常、shutdown 和普通 timeout，都必须经过 FINALIZE。

## 修复副本实现

文件：`self_code_lifecycle_fixed.py`

1. `scan_unclaimed_tasks()` 返回 `Task` 对象，不再混用字符串和字典。
2. 自动认领消息注入 task ID、subject、description。
3. 删除 S16 遗留的无限 idle wait，建立显式 WORK -> IDLE 状态机。
4. WORK 最多 10 轮，防止无限工具循环。
5. `claim_task()` 增加 owner 校验，并在进程内锁中完成读、检查、写。
6. `complete_task()` 可校验调用者是否为任务 owner。
7. teammate 结束时自动将未完成 owned task 重新置为 pending。
8. 启动时恢复没有活跃 owner 的 orphaned in_progress task。
9. `active_teammates` 注册和移除使用锁，清理放在 finally。
10. MessageBus 使用每 inbox 锁和 `os.replace()` 原子 drain，避免 read + unlink 丢消息。
11. 增加 `.agent_logs/<agent>.jsonl` 结构化生命周期日志。
12. 修正启动横幅为 `s17_autonomous_agent_loop`。

## 日志事件

关键事件包括：

- `thread_dispatched`, `thread_started`, `thread_stopped`
- `work_entered`, `llm_turn_started`, `llm_turn_finished`
- `idle_entered`, `idle_to_work`, `idle_timeout`, `inbox_wakeup`
- `task_claimed`, `task_completed`, `tasks_requeued`, `orphan_recovered`
- `llm_error`, `tool_error`, `thread_crashed`, `mailbox_decode_error`

定位卡死时优先检查：最后一条事件、active registry、task owner/status、inbox mtime。

## 验证结果

`test_lifecycle_fixed.py` 的 9 项无网络测试全部通过：

1. IDLE 自动认领并注入完整任务。
2. owner 防抢占及 complete owner 校验。
3. inbox 消息唤醒。
4. shutdown request 即时响应。
5. WORK -> IDLE -> 自动认领 -> WORK -> complete -> timeout 完整循环。
6. LLM 错误清理 active registry。
7. LLM 错误释放 owned in_progress task。
8. 启动时恢复 orphan task。
9. 100 条并发 inbox 消息发送/消费零丢失。

真实模型最小集成也成功：首轮只回复 READY，Harness 在 IDLE 自动认领任务，模型写入 `lifecycle_probe.txt`、调用 `complete_task`，最终任务为 `completed`、owner 为 probe、active registry 已清空。测试命令末尾出现的 Windows 临时目录清理错误，是测试进程仍把临时目录作为当前目录造成的，不影响 Agent 流程结果。

## 运行方式

```powershell
python .\s17_autonomous_agents\test_lifecycle_fixed.py
```

交互运行副本：

```powershell
cd .\s17_autonomous_agents
python .\self_code_lifecycle_fixed.py
```

## 仍需注意的生产化边界

- 当前 task/mailbox 锁只覆盖单进程线程；多个 Agent 进程需要 lockfile、SQLite 或事务数据库。
- `recover_orphaned_tasks()` 假设单进程。多进程环境应使用 lease、worker ID、heartbeat 和 `lease_expires_at`，不能仅凭本进程 active registry 判断 owner 已死亡。
- 异常任务反复 requeue 可能形成 poison loop；应增加 `attempt_count`、`last_error`、最大重试次数和 dead-letter 状态。
- MessageBus 遇到损坏 JSON 行目前记录并跳过；生产系统应将原始 processing 文件移到 quarantine。
- 应为 task 状态增加 `updated_at`、`claimed_at`、`heartbeat_at`、`finished_at`，便于判断是真卡死还是仍在运行。
- plan approval 若属于安全边界，必须在工具分发层强制 gating，不能只依赖提示词。
- daemon thread 在进程强制退出时不会运行 finally；可靠常驻 Agent 应使用可恢复的持久队列和非 daemon worker，配合显式 shutdown/join。

## 原始现场任务的处理建议

不要直接把所有 `in_progress` 标成 completed。应先确认产物是否存在：

- 有有效产物：由对应 owner 或 Lead 验证后 complete。
- 无产物且 owner 已死亡：清空 owner，恢复为 pending，再由新 teammate 认领。
- 多次失败：记录 last_error 并进入 failed/dead-letter，避免无限重试。