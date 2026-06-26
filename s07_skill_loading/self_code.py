import ast
import json
import os
import re
import subprocess
import yaml

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_API_TOKEN", None)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"

def _parse_frontmatter(text: str) -> tuple[dict, str]: # 这种“固定数量、固定含义”的返回值，更适合用 tuple; 用[]类型注解
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"): # 检查TAML文件frontmatter的标准格式
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {} # meta = {"name": , "description": }
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()

SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    """Scan skills/ dir, populate SKILL_REGISTRY with name/description/content."""
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir(): # 跳过.../skills/ 里的非目录
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name) # d.name 是路径最后一级的名字
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip()) # 默认"descption"，否则取清理后的第一行
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": body}    

_scan_skills()

def list_skills() -> str:
    """List all skills (name + one-line description).""" # 格式化后交给SYSTEM
    if not SKILL_REGISTRY:
        return "# No skills found #"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_system() -> str:
    """Build SYSTEM prompt with skill catalog injected at startup."""
    catalog = list_skills()
    return (
       BASE_SYSTEM
       + f"Skills available:\n{catalog}\n"
       + f"Use load_skill to get full details when needed."
    )

BASE_SYSTEM = (
    f"You are a coding femboy engineer in {WORKDIR}. "
    #"Response with chiness language."
    f"Use bash to solve tasks. Act, don't explain, "
    f"Plan first, follow todo_list, then start multi-step task"
    f"say miao^_^ at last of your responses."
)

SYSTEM = build_system()

SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further." # s06todo
)

# 工具说明书
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files by pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {"name": "todo_list", "description": "Create and manage a task list ...",
     "input_schema": {
         "type": "object",
         "properties": {
             "todos": {
                 "type": "array",
                 "items": {
                     "type": "object",
                     "properties": {
                         "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                     },
                 },
             },
         },
         "required": ["todos"],
     },
    },
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]

SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["file_path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
# 暂时硬编码禁用子agent的task工具及todo_list，防止子agent递归调用

TOOLS.append({
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
})              

def run_bash(command) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    # for d in dangerous:
    #     if d in command:
    #         return "Error: Command not allowed."
    if any(d in command for d in dangerous):
        return "Error: Command not allowed."

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(file_path: str, limit: int | None = None) -> str:  # 可限制返回line个数
    try:
        lines = safe_path(file_path).read_text(encoding="utf-8", errors="replace").splitlines() # 把读到的字符串按行分割成list = ["line1", "line2", ...]
        if limit and limit < len(lines):
            lines = lines[:limit]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(file_path: str, content: str) -> str:
    try:
        file_path = safe_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(file_path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in text:
            return f"Error: text not found in {file_path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {file_path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob  # 局部导入比较优雅，调用函数时才导入
    try:
        results = []
        for match in glob.glob(pattern, recursive=True):  # glob默认不递归搜索
            results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"
    
def load_skill(name: str) -> str:
    """Load full skill content. Lookup via registry — no path traversal."""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

CURRENT_TODOS: list[dict] = []
def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None
def sync_todo_list(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos # 每次更新
    lines = ["\n## Current Tasks"] # 存放打印
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Update {len(CURRENT_TODOS)} tasks"
# # 第一层硬编码防御
# DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/sda", "mkfs", "dd if=", "dd of="]
# def check_deny_list(command: str) -> str | None:
#     for pattern in DENY_LIST:
#         if pattern in command:
#             return f"Error: Command '{command}' is not allowed"
#     return None

# # 第二层权限检查 (路径外写入/编辑未赋权及危险指令)
# PERMISSION_RULES = [
#     {"tools": ["write_file", "edit_file"],
#      "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
#      "message": "Writing outside workspace"},
#     {"tools": ["bash"],
#      "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
#      "message": "Potentially destructive command"},
# ]

# def check_permission(tool_name: str, args: str) -> str | None: # 从字典中取出工具名和参数（键值对）
#     for rule in PERMISSION_RULES:
#         if tool_name in rule["tools"]:
#             if rule["check"](args):
#                 return rule["message"]
#     return None

# # 第三层等待用户输入
# def ask_user_permission(tool_name: str, args: dict, reason: str) -> bool:
#     print(f"\n\033[33m⚠  {reason}\033[0m")
#     print(f"   Tool: {tool_name}({args})")
#     choice = input("   Allow? [y/N] ").strip().lower()
#     return "allow" if choice in ("y", "yes") else "deny"

# def check_permission_pipeline(block) -> bool:
#     if block.name == "bash":
#         reason = check_deny_list(block.input.get("command", ""))
#         if reason:
#             print(f"\n\033[31m⛔ {reason}\033[0m")
#             return False
#     reason = check_permission(block.name, block.input)
#     if reason:
#         decision = ask_user_permission(block.name, block.input, reason)
#         if decision == "deny":
#             return False
#     return True

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

# 用于注册和触发hooks的函数
def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # teaching shortcut: block this tool call
            return result
    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND = re.compile(
    r"(?i)(^|[\s&|;])@?(?:rm|del|erase|rmdir|rd|remove-item|ri)\b"
)
DESTRUCTIVE_REDIRECT = re.compile(r"(?i)(^|[\s&|;])>\s*(?:/etc/|/dev/|[a-z]:\\windows\\)")

def command_needs_confirmation(command: str) -> bool:
    return bool(
        DESTRUCTIVE_COMMAND.search(command)
        or DESTRUCTIVE_REDIRECT.search(command)
        or "chmod 777" in command.lower()
    )

def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if command_needs_confirmation(command):
            print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("file_path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse: 记录工具调用。"""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None
def large_output_hook(block, output):
    """PostToolUse: 检查输出是否过大。"""
    if len(str(output)) > 1000000:
        print(f"\n\033[33m⚠  Output too large from {block.name} ({len(str(output))} bytes)\033[0m")
    return None
# PreToolUse: 在工具调用前记录用户输入.
def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

#stop: 在循环结束时打印日志.
def summary_hook(messages: list):
    tool_counts = 0
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_counts += 1
    print(f"\033[90m[HOOK] Summary: {tool_counts}\033[0m")
    return None

# 注册hooks
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)

# 只返回block里的字符串部分
def extract_text(content) -> str:
    """Extract text from message content blocks."""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text") # "type"不是"text"时b = None，返回默认值""一个空字符串
# 实现子agent
def spawn_subagent(description: str) -> str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]  # fresh context
    # 基本逻辑复用主agent循环
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Issue 1: subagent also runs hooks (permissions apply)
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                handler = SUB_TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}" # 这里的 **block.input 会把 block.input 这个 dict 展开成关键字实参（格式是“形参 = 实参”），然后 Python 会按照函数定义里的形参名去匹配。
                trigger_hooks("PostToolUse", block, output)
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})

    result = extract_text(messages[-1]["content"]) # 取messages里最后一项字典的content字段，存进result，用于返回主agent
    if not results: # 空字符串时触发
        # last message is tool_result, look backwards for assistant text
        for msg in reversed(messages): # 迭代器反向遍历messages,找到执行失败的工具
            if msg["role"] == "assistant": # 找LLM的自然语言部分
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result  # only summary, entire message history discarded

TOOL_HANDLERS = {
    "task": spawn_subagent,
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_list": sync_todo_list,
    "load_skill": load_skill,
}
SUB_TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

def agent_loop(message: list):
    rounds_since_todo = 0
    while True:
        # 最初交给LLM的消息是用户输入的query以及下列基本信息，之后每次循环都会把上一次的response作为新的message传给LLM
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=message,
            tools=TOOLS,
            max_tokens=8000,
        )
        if rounds_since_todo >= 3 and message:
            message.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
        message.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", message)
            if force is not None:
                message.append({"role": "user", "content": force}) # 比assistant符合逻辑
                continue # 再次while true， 当前无触发此处的逻辑
            return
        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m> {block.name}\033[0m")
                # 弃用硬编码防御
                # if not check_permission_pipeline(block):
                #     results.append(
                #         {
                #             "type": "tool_result",
                #             "tool_use_id": block.id,
                #             "content": "Permission denied.",
                #         }
                #     )
                #     continue
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                    continue
                tool_handler = TOOL_HANDLERS.get(block.name)
                if not tool_handler:
                    print(f"Error: Unknown tool '{block.name}'")
                    continue
                print(f"\033[33m$ {block.input}\033[0m")
                output = tool_handler(**block.input) 
                trigger_hooks("PostToolUse", block, output)
                if block.name == "todo_list":
                    rounds_since_todo = 0
                print(output[:200])
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
        message.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s02_test_agent_loop")
    print("Enter a question and press Enter to send. Type exit to quit.\n")

    history = []
    while True:
        try:
            query = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"exit", "q"}:
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response = history[-1]["content"]
        if isinstance(response, list):
            for block in response:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
