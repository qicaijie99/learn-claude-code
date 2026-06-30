先给结论：**s08 / s09 / s10 参考答案都是“教学阶段代码”，每一版只突出一个主题。我的优化版不是要否定参考答案，而是把你从 s08、s09 继承来的功能合并到 s10 时，把边界重新整理清楚。**

参考答案的设计顺序是：

```text
s08：只讲上下文压缩
s09：只讲 memory
s10：只讲 system prompt 动态组装和缓存
```

你的代码是：

```text
s08 + s09 + s10 + hooks + skills + subagent + todo_list
```

所以优化版比 s10 参考答案“更好”的前提是：**它更适合你的综合版本，不是更适合教学最小版。**

---

# 1. s10 参考答案是不是和 memory 解耦？

**是，但不是完全不管 memory。**

准确说：

```text
s10 参考答案把 memory 的“读取判断”放进 update_context，
把 memory 的“提示词渲染”放进 assemble_system_prompt，
agent_loop 只使用 get_system_prompt(context)。
```

这叫**职责解耦**，不是功能删除。

s09 参考答案自己的注释说得很清楚：memory 流程包括把 `MEMORY.md` 索引放进 system prompt、选择相关 memory 注入内容、压缩、结束后提取新 memory、定期 consolidate。

而 s10 的教学重点变成：

```text
context 反映真实状态
system prompt 由 context 动态组装
get_system_prompt 做缓存
```

所以 s10 不是“不要 memory”，而是把 memory 当成 `context` 里面的一部分。

---

# 2. 参考答案本身为什么简单？

因为 s10 参考答案只演示 system prompt assembly。

它的 memory 逻辑大概是：

```python
if MEMORY_INDEX.exists():
    content = MEMORY_INDEX.read_text().strip()
    memories = content
```

也就是只读 `.memory/MEMORY.md` 的内容。

这对教学足够，因为它只想证明：

```text
真实状态变化 → context 变化 → system prompt 重新组装
```

但它没有完整继承 s09 的：

```text
select_relevant_memories()
load_memories()
extract_memories()
consolidate_memories()
```

你的代码继承了 s09 的完整 memory 系统，所以不能只照搬 s10 最小参考答案。

---

# 3. 我优化的核心原则

我把你的综合版收敛成这条链路：

```text
messages：只保存真实对话历史
context：保存当前运行状态快照
system prompt：由 context 渲染出来
memory：通过 context 注入 system，不再偷塞进 user message
agent_loop：只负责循环、工具执行、压缩、刷新 context
```

这比你老代码清楚，因为你老代码里同时存在两条 memory 注入路径：

```text
路径 A：update_context() 读取 MEMORY.md 索引 → context["memories"] → system prompt
路径 B：load_memories() 读取完整 memory → 拼到最新 user.content 前面
```

这会导致“memory 一部分在 system，一部分在 user message”，后续压缩、缓存、memory extraction 都容易混乱。

---

# 4. 优化位置一：`PROMPT_SECTIONS` 只保留静态片段

## 你的老代码 / s10 参考答案的问题

你的老代码里：

```python
PROMPT_SECTIONS = {
    "identity": "...",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
}
```

问题是你的真实工具并不只有：

```text
bash
read_file
write_file
```

你的综合版还有：

```text
edit_file
glob
todo_list
load_skill
compact
task
```

s08 参考答案里也确实有更多工具，包括 `todo_write`、`task`、`load_skill`、`compact` 等。

所以如果 system prompt 里写死：

```text
Available tools: bash, read_file, write_file.
```

但 API 的 `tools=TOOLS` 里实际有更多工具，就会出现：

```text
模型看到的工具说明
≠
实际可用工具列表
```

## 优化版为什么更好

优化版改成：

```text
PROMPT_SECTIONS 只保存 identity / memory_policy 这种稳定文本
tools / workspace / skills / memory 都从 context 取
```

也就是：

```text
真实 TOOL_HANDLERS 有什么
  ↓
update_context()
  ↓
context["enabled_tools"]
  ↓
assemble_system_prompt()
  ↓
system prompt 里的工具列表
```

这样不会出现“真实工具已经变了，但 system prompt 还写旧工具”的问题。

---

# 5. 优化位置二：`assemble_system_prompt()` 不再吃硬编码，而是吃 context

## 参考答案的优点

s10 参考答案的关键思想是：

```text
assemble_system_prompt(context)
```

也就是提示词不是写死成一个全局 `SYSTEM`，而是根据当前状态组装。

这比 s08 的：

```python
SYSTEM = build_system()
```

更动态。s08 里 `SYSTEM` 是启动时构造一次的，后面 agent_loop 直接用这个固定 `SYSTEM`。

## 你的老代码对应问题

你的老代码虽然引入了：

```python
get_system_prompt(context)
```

但 `assemble_system_prompt()` 里面很多内容还是从 `PROMPT_SECTIONS` 固定取，不是真正从 `context` 取。

这就形成一个尴尬状态：

```text
context 里有 enabled_tools / workspace
但 assemble_system_prompt 没真正使用它们
```

## 优化版为什么更好

优化版的 `assemble_system_prompt()` 按这个顺序拼：

```text
identity
workspace
enabled_tools
skills
memory_index
relevant_memories
```

这样 `context` 才真正成为 system prompt 的唯一数据源。

---

# 6. 优化位置三：memory 分成 `memory_index` 和 `memories`

这是最重要的改动。

## s09 参考答案的设计

s09 参考答案本来就是两层 memory：

```text
MEMORY.md：索引
.memory/xxx.md：完整 memory 文件
```

它的 `load_memories()` 会先选相关文件，再读取完整 memory 内容，最后包进 `<relevant_memories>`。

所以 s09 的设计其实是：

```text
索引：便宜，常驻
正文：昂贵，只注入相关的
```

这是好的。

## 你的老代码问题

你的 `update_context()` 只有：

```python
"memories": MEMORY_INDEX.read_text()
```

也就是说：

```text
context["memories"] = MEMORY.md 索引
```

但你的 `agent_loop()` 又另外做：

```python
memories_content = load_memories(message)
```

然后把完整 memory 拼进最新用户消息。

这就导致：

```text
system prompt 里：memory 索引
user message 里：完整 memory 正文
```

memory 被拆成两条通道了。

## 优化版为什么更好

优化版改成：

```python
context = {
    "memory_index": read_memory_index(),
    "memories": load_memories(messages),
}
```

含义非常清楚：

| 字段             | 作用                  |
| -------------- | ------------------- |
| `memory_index` | 低成本目录，让模型知道有哪些长期记忆  |
| `memories`     | 本轮真正相关的完整 memory 正文 |

然后它们都由 `assemble_system_prompt()` 注入 system prompt。

这样 memory 的生命链路统一成：

```text
.memory/*.md
  ↓
MEMORY.md index
  ↓
load_memories(messages)
  ↓
context
  ↓
system prompt
  ↓
model
```

不再绕到 user message 里。

---

# 7. 优化位置四：不再改写最新 user message

## s09 参考答案为什么这么写？

s09 参考答案里这段：

```python
memories_content = load_memories(messages)
memory_turn = len(messages) - 1 ...
...
request_messages[memory_turn] = {
    **messages[memory_turn],
    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
}
```

它是为了快速实现：

```text
把相关 memory 内容临时塞进当前用户问题前面
```

这在 s09 教学版里可以接受，因为它当时还没有 s10 的 runtime system prompt assembly。相关代码就在 s09 的 agent_loop 里，先加载 memory，再临时替换 `request_messages[memory_turn]["content"]`。

## 你的综合版为什么不应该继续这样做？

因为到了 s10，你已经有了：

```text
context -> get_system_prompt(context) -> system
```

如果还把 memory 塞进 user message，就会变成：

```text
memory 既是系统背景，又像用户当前说的话
```

这有几个问题：

1. **语义混乱**：memory 是背景，不是用户本轮输入。
2. **压缩混乱**：压缩 messages 时，memory 可能被当作用户原话总结进去。
3. **提取混乱**：`extract_memories()` 可能把旧 memory 再当成新对话提取，造成重复。
4. **缓存混乱**：system prompt 缓存看不到 user message 被临时改写的那部分。
5. **调试混乱**：history 里没有 memory，但 request_messages 里有 memory，实际发给模型的内容和你打印/保存的不一致。

## 优化版为什么更好

优化版直接删掉“改写 user message”的路线。

最终变成：

```text
messages = 真实对话
context["memories"] = 相关 memory
system = assemble_system_prompt(context)
```

这就是 s10 思想的完整落实。

---

# 8. 优化位置五：`get_system_prompt()` 的缓存才真正可靠

## 老代码的问题

你老代码里的缓存逻辑本身是对的：

```python
key = json.dumps(context, sort_keys=True, ...)
if key == _last_context_key:
    return _last_prompt
```

但问题在于：

```text
context 不完整 / assemble_system_prompt 不完全使用 context
```

于是会出现两种不一致：

### 情况 A

```text
context 变化了
但 assemble 出来的 prompt 实际没变化
```

比如 `enabled_tools` 变了，但 `assemble_system_prompt()` 不用它。

### 情况 B

```text
真实 prompt 应该变化
但 context 没变化
```

比如你偷偷把 memory 拼进 `request_messages[memory_turn]`，这不会体现在 system prompt 的 cache key 里。

## 优化版为什么更好

优化版让所有会影响 system prompt 的东西都进入 context：

```text
workspace
enabled_tools
skills
memory_index
memories
```

这样：

```text
context key 变化
≈
system prompt 应该变化
```

缓存语义就对齐了。

---

# 9. 优化位置六：context 刷新时机更完整

## s08 / s09 参考答案的情况

s08 的 agent_loop 每次 LLM 调用前会跑压缩管线：

```python
messages[:] = tool_result_budget(messages)
messages[:] = snip_compact(messages)
messages[:] = micro_compact(messages)
if estimate_size(messages) > CONTEXT_LIMIT:
    messages[:] = compact_history(messages)
```

这说明：**messages 在 LLM 调用前会被改写。**

s09 又在一轮结束后执行：

```python
extract_memories(pre_compress)
consolidate_memories()
```

也就是一轮结束后 memory 文件可能变化。

## 你的老代码问题

你虽然在工具轮结束后刷新了：

```python
context = update_context(context, message)
system = get_system_prompt(context)
```

但在以下情况里还不够统一：

```text
进入 agent_loop 前
auto compact 后
追加 reminder 后
memory extraction 后
```

这些都会影响 context 或 prompt 的合理状态。

## 优化版为什么更好

优化版做成：

```text
进入 agent_loop：先 update_context
auto compact 改写 messages 后：update_context
追加 reminder 后：update_context
每个 tool round 后：update_context
agent_loop 返回后：主程序再 update_context
```

这样 `system` 更接近真实状态快照。

---

# 10. 优化位置七：修复 compact 工具的 tool_use / tool_result 配对风险

这是 s08 参考答案本身就留下的教学版隐患。

## s08 参考答案的问题

s08 中遇到 `compact` 工具时，逻辑是：

```python
if block.name == "compact":
    messages[:] = compact_history(messages)
    results.append({"type": "tool_result", ...})
    messages.append({"role": "user", "content": results})
    break
```

问题在于：

```text
messages[:] = compact_history(messages)
```

会把历史整体替换成 summary。

但此时刚刚的 assistant 消息里包含 `tool_use`，它应该紧跟一个对应的 user `tool_result`。

如果你先把 assistant `tool_use` 压缩掉，再追加 user `tool_result`，就可能变成：

```text
user: [Compacted summary]
user: tool_result
```

而不是：

```text
assistant: tool_use
user: tool_result
```

s08 参考答案确实是在 `compact` 分支里先 compact，再 append tool_result。

## 优化版为什么更好

优化版改成：

```text
看到 compact 工具
  ↓
先记录 compact_after_tool_round = True
  ↓
仍然 append user tool_result
  ↓
tool_use / tool_result 配对完成
  ↓
再 compact_history(messages)
```

也就是：

```text
先满足 API 协议
再压缩上下文
```

这比参考答案更适合真实运行。

---

# 11. 优化位置八：unknown tool 也返回 tool_result

## 你的老代码问题

你老代码里如果工具不存在：

```python
if not tool_handler:
    print(f"Error: Unknown tool '{block.name}'")
    continue
```

问题是：模型已经发出了一个 `tool_use`，你却没有给它对应的 `tool_result`。

这会导致下一轮 API 看到：

```text
assistant 有 tool_use
但 user 没有对应 tool_result
```

协议上不完整。

## 优化版为什么更好

优化版改成：

```python
if not tool_handler:
    results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": f"Unknown tool: {block.name}",
    })
    continue
```

也就是：即使工具未知，也要把错误作为 tool_result 返回给模型。

---

# 12. 优化位置九：subagent 的 `results` 未定义隐患

这个不是 s08 / s09 参考答案的问题，主要是你老代码里的问题。

你的老代码里有这种结构：

```python
for _ in range(30):
    response = ...
    if response.stop_reason != "tool_use":
        break
    results = []
    ...
if not results:
    ...
```

如果 subagent 第一次回复就是自然语言，没有工具调用，那么：

```text
results 根本没定义
```

后面 `if not results` 就会报：

```text
UnboundLocalError
```

优化版加了：

```python
results = []
```

放在循环之前。

这样无论 subagent 是否调用工具，后面都安全。

---

# 13. 优化位置十：memory extraction 加入最终 assistant 回复

## s09 参考答案的做法

s09 在每轮 while 开头保存：

```python
pre_compress = ...
```

然后最终结束时：

```python
extract_memories(pre_compress)
```

这保留了压缩前信息，目的是避免压缩后 history 信息损失。

这个设计有道理。

## 但它的不足

`pre_compress` 是在本轮模型最终回复之前保存的，所以它可能不包含：

```text
最终 assistant 回复
最终结论
最终项目事实判断
```

如果你只想提取用户偏好，影响不大。

但你的 memory 类型里有：

```text
project
reference
feedback
```

这些有可能来自 assistant 最终整理出的结论。

## 优化版为什么更好

优化版用：

```python
extraction_source = pre_compress + [{"role": "assistant", "content": response.content}]
extract_memories(extraction_source)
```

这样既保留了：

```text
压缩前原始对话
```

又包含：

```text
最终 assistant 回复
```

对综合版更稳。

---

# 14. 优化位置十一：memory 失败不再静默吞掉

## s09 参考答案的问题

s09 很多地方是：

```python
except Exception:
    pass
```

比如 memory 选择、提取、合并失败时，它会静默失败。s09 的 memory 提取函数确实在调用 LLM、解析 JSON 后，最后用 `except Exception: pass` 兜底。

教学版这样可以避免程序崩。

但你调试综合版时，这会很痛苦。

## 优化版为什么更好

优化版改成打印：

```text
[Memory selection failed] ...
[Memory extraction failed] ...
[Memory consolidation failed] ...
```

这样你至少知道：

```text
是没有提取出 memory
还是 API / JSON / 文件写入出错
```

这对学习和调试更友好。

---

# 15. 优化位置十二：保留 s08 的压缩顺序，但把它纳入 s10 链路

s08 的压缩顺序是：

```text
tool_result_budget → snip_compact → micro_compact → auto compact
```

参考答案明确写了“cheap first, expensive last”，执行顺序是 `budget → snip → micro → auto`。

我的优化版没有改这个核心顺序。

但我补了一件 s10 需要的事：

```text
只要 messages 被 compact 改写
就刷新 context 和 system prompt
```

因为 s10 的 system prompt 依赖 context，而 context 又可能依赖 messages：

```text
messages 变化
  ↓
load_memories(messages) 选择结果可能变化
  ↓
context["memories"] 可能变化
  ↓
system prompt 可能变化
```

所以 s08 的压缩机制接入 s10 后，必须重新考虑 context 刷新。

---

# 16. 对照表：参考答案、你的老代码、优化版

| 位置                  | 参考答案                 | 你的老代码问题                           | 优化版                                 |
| ------------------- | -------------------- | --------------------------------- | ----------------------------------- |
| s08 压缩              | 正确演示四层压缩             | 继承了，但 compact 分支仍有协议隐患            | 保留顺序，修复 compact 配对                  |
| s09 memory index    | `MEMORY.md` 进 system | 你也读 index，但放进 `memories` 字段，语义混   | 分成 `memory_index`                   |
| s09 relevant memory | 拼进当前 user message    | 你也这样做，和 s10 context 注入重复          | 放进 `context["memories"]`，统一进 system |
| s10 prompt assembly | 用 context 组装 system  | 你有 context，但 tools/workspace 仍硬编码 | 所有动态信息都从 context 渲染                 |
| prompt cache        | 根据 context JSON 判断   | context 和真实 prompt 没完全对齐          | context 包含所有 prompt 输入              |
| agent_loop          | s08/s09/s10 各自突出一个主题 | 多主题混合，边界乱                         | agent_loop 只循环，状态交给 context         |
| unknown tool        | 参考答案较简化              | 可能不返回 tool_result                 | 一律返回 error tool_result              |
| subagent            | 参考答案里相对简化            | 你的版本有 `results` 未定义隐患             | 预先初始化                               |
| memory extraction   | 从 pre_compress 提取    | 可能漏最终 assistant 回复                | pre_compress + final assistant      |
| debug               | 多处 `pass`            | 失败无感知                             | 打印失败原因                              |

---

# 17. 最重要的架构区别

你的老代码更像这样：

```text
agent_loop
  ├─ 负责压缩
  ├─ 负责 memory 读取
  ├─ 负责改写 user message
  ├─ 负责 prompt 刷新
  ├─ 负责 tool 执行
  ├─ 负责 memory 写入
  └─ 负责 context 更新
```

优化后是：

```text
memory 模块
  ├─ read_memory_index
  ├─ load_memories
  ├─ extract_memories
  └─ consolidate_memories

context 模块
  └─ update_context：把真实状态汇总成快照

prompt 模块
  ├─ assemble_system_prompt
  └─ get_system_prompt

agent_loop
  ├─ 压缩 messages
  ├─ 调模型
  ├─ 执行工具
  ├─ 保证 tool_use/tool_result 配对
  └─ 在关键节点刷新 context/system
```

这就是我说它“比直接照搬参考答案更适合你的综合代码”的原因。

---

# 18. 一句话总括

s08 参考答案解决的是：

```text
messages 太长怎么办
```

s09 参考答案解决的是：

```text
长期 memory 怎么存、怎么选、怎么写回
```

s10 参考答案解决的是：

```text
system prompt 怎么由 runtime context 动态组装并缓存
```

你的代码要做的是三者合体，所以优化版的核心价值是：

```text
让 memory、context、system prompt、messages、压缩管线各归各位。
```

最关键的改动就是：

```text
不要再把 memory 拼进 user.content；
把 memory 作为 context 的一部分，统一注入 system prompt。
```

优化代码仍在这里：

[下载优化后的代码](sandbox:/mnt/data/optimized_agent_s10_memory.py)
