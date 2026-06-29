import ast
import json
import os
import re
import subprocess
import yaml, time

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
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

MEMORY_DIR = WORKDIR / ".memory"; MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_TYPES = ["user", "feedback", "project", "reference"]
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

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

def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """Write a single memory file with YAML frontmatter."""
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
    )
    _rebuild_index()
    return filepath

def _rebuild_index():
    """Rebuild MEMORY.md index from all memory files."""
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem) # dict.get:如果字典里有这个 key就取对应值；如果没有就用默认值。此处意为：优先取 frontmatter 里的 name；如果没有就用文件名去掉后缀。
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")

def read_memory_index() -> str:
    """Read MEMORY.md index (injected into SYSTEM every turn)."""
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""

def read_memory_file(filename: str) -> str | None:
    """Read a single memory file's full content."""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()

# 扫描所有 memory 文件，返回一个列表
def list_memory_files() -> list[dict]:
    """List all memory files with metadata."""
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result

def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """Select relevant memory filenames by matching recent conversation against
    memory names/descriptions. Uses a simple LLM call (or falls back to keyword
    matching on name+description)."""
    files = list_memory_files()
    if not files:
        return []
     # Collect recent user text for context
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for b in content:
                    if getattr(b, "type", None) == "text": content = " ".join(str(getattr(b, "text", "")))
            if isinstance(content, str): recent_texts.append(content)
            if len(recent_texts) >= 3: break
    recent = " ".join(reversed(recent_texts))[:2000]        
                
    if not recent.strip():
        return []
    
    # Build catalog of name + description for LLM to choose from
    catalog_lines = []
    for i, f in enumerate(files):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)
    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = extract_text(response.content).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', text, re.DOTALL)  # 抠出JSON array [..., ... ]
        if match:
            indices = json.loads(match.group())  # 把 JSON 格式的字符串，解析成 Python 对象。
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx <len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items: break
            return selected
    except Exception:
        pass            

    # Fallback: keyword matching on name + description
    keywords = [w.lower() for w in recent.split() if len(w) > 3]  # 把最近对话按空格切词，只保留长度大于 3 的词，并转成小写。
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected    

# 选择相关 memory 文件，然后读取它们的完整内容，拼成一个字符串，准备注入上下文。
def load_memories(messages: list) -> str:
    """Load relevant memory content for injection into context."""
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""

    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def extract_memories(messages: list):
    """Extract new memories from recent dialogue. Runs after each turn."""
    # Collect recent conversation text
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return
    
    # Check existing memories to avoid duplicates(目的是避免重复写 memory。它是一种“用少量 token 换长期记忆质量”的设计。)
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800
        )
        text = extract_text(response.content).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass

CONSOLIDATE_THRESHOLD = 10

def consolidate_memories():
    """Merge duplicate/stale memories. Triggered when file count ≥ threshold."""
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    # 合并重复记忆；
    # 删除过时或被新信息否定的记忆；
    # 总数控制在 30 条以内；
    # 优先保留重要用户偏好。
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )
    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000
        )
        text = extract_text(response.content).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # Remove old memory files (keep MEMORY.md)
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body) #  内部调用 _rebuild
            print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
    except Exception:
        pass


# Build SYSTEM with memory index（构造系统提示词，把 memory 索引塞进 SYSTEM。）
def build_system() -> str:
    """Build SYSTEM prompt with skill catalog injected at startup."""
    catalog = list_skills()
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        BASE_SYSTEM
        + f"Skills available:\n{catalog}\n"
        + f"Use load_skill to get full details when needed."
        + f"You are a coding agent at {WORKDIR}."
        + f"{memories_section}\n"
        + "Relevant memories are injected below. Respect user preferences from memory.\n"
        + "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )

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
     {"name": "compact", "description": "Summarize earlier conversation to free context space.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
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

# hook 决定某些代码在 agent 生命周期的哪个节点自动执行
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

CONTEXT_LIMIT = 50000; KEEP_RECENT = 3; PERSIST_THRESHOLD = 30000
def estimate_size(msgs): return len(str(msgs))

def _block_type(block):
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _message_has_tool_use(msg):
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg):
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result"
               for block in content)

# L1: snipCompact — trim middle messages
def snip_compact(messages, max_messages=50):
    if len(messages) <= max_messages: return messages
    keep_head, keep_tail = 3, max_messages - 3
    head_end, tail_start = keep_head, len(messages) - keep_tail
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]


# L2: microCompact — old result placeholders
def collect_tool_results(messages):
    blocks = []
    for mi, msg in enumerate(messages): # 枚举出 mi: message 的下标, msg: 当前 message 本身
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks

def micro_compact(messages):
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT: return messages
    for _, _, block in tool_results[:-KEEP_RECENT]: 
        # 从开头取到“倒数第 KEEP_RECENT 个”之前
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


# L3: toolResultBudget — persist large results to disk
def persist_large_output(tool_use_id, output):
    if len(output) <= PERSIST_THRESHOLD: return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists(): path.write_text(output) # 写入content
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

def tool_result_budget(messages, max_bytes=200_000):
    last = messages[-1] if messages else None # 取倒数第一个message
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return messages
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"] # 列表推导式
    # 等价于
    # for i, b in enumerate(last["content"]):
    # if isinstance(b, dict) and b.get("type") == "tool_result":
    #     blocks.append((i, b))
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes: return messages
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True) # sort(): 改原列表, sorted(): 生成新列表
    for _, block in ranked:
        if total <= max_bytes: break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD: continue
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


# L4: autoCompact — LLM full summary
def write_transcript(messages):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages: f.write(json.dumps(msg, default=str) + "\n")
    return path

def summarize_history(messages):
    conversation = json.dumps(messages, default=str)[:80000] # 隐患：有效信息缺失。可以试试首尾
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    return "\n".join(
        getattr(block, "text", "") # 取对象属性
        for block in response.content
        if getattr(block, "type", None) == "text").strip() or "(empty summary)"

def compact_history(messages):
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# Emergency: reactiveCompact — on API error
def reactive_compact(messages):
    transcript = write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    summary = summarize_history(messages[:tail_start])
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]

# 主动压缩：compact_history，全量替换成 summary
# 紧急压缩：reactive_compact，总结旧历史 + 保留最近尾部

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

MAX_REACTIVE_RETRIES = 1  # retry limit for reactive compact

def agent_loop(message: list):
    rounds_since_todo = 0
    reactive_retries = 0
    memories_content = load_memories(message)
    memory_turn = len(message) - 1 if message and isinstance(message[-1].get("content"), str) else None
    # s09: build system once per user turn; memory is updated after the loop returns
    system = build_system()

    while True:

        # s09: save pre-compression snapshot for accurate memory extraction
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role",""),
            "content": str(m.get("content",""))} for m in message]
        
        # s08 change: three preprocessors (0 API calls, cheap first)
        # Order matches CC source: budget → snip → micro
        message[:] = tool_result_budget(message)    # L3: persist large results first
        message[:] = snip_compact(message)          # L1: trim middle
        message[:] = micro_compact(message)         # L2: old result placeholders

        # s08 change: tokens still over threshold → LLM summary (1 API call)
        if estimate_size(message) > CONTEXT_LIMIT:
            print("[auto compact]")
            message[:] = compact_history(message)
        # 最初交给LLM的消息是用户输入的query以及下列基本信息，之后每次循环都会把上一次的response作为新的message传给LLM
        if rounds_since_todo >= 3 and message:
            message.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        try:
            request_messages = message
            if memories_content and memory_turn is not None and memory_turn < len(message):
                request_messages = message.copy()
                request_messages[memory_turn] = {
                    **message[memory_turn],
                    "content": memories_content + "\n\n" + message[memory_turn]["content"], # 后"content"覆盖前者(就能保留其他字段，只替换 content)
                }
            response = client.messages.create(
                model=MODEL,
                system=system,
                messages=request_messages,
                tools=TOOLS,
                max_tokens=8000,
            )
            reactive_retries = 0  # reset on successful API call  
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                message[:] = reactive_compact(message)
                reactive_retries += 1
                continue
            raise

        message.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
        #     force = trigger_hooks("Stop", message)
        #     if force is not None:
        #         message.append({"role": "user", "content": force}) # 比assistant符合逻辑
        #         continue # 再次while true， 当前无触发此处的逻辑
            # s09: extract from pre-compression snapshot for full fidelity
            extract_memories(pre_compress)
            consolidate_memories()
            return
        
        rounds_since_todo += 1
        results = []
        # 跳过text block
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            # s08: compact tool triggers compact_history, not a no-op string
            if block.name == "compact":
                message[:] = compact_history(message)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                message.append({"role": "user", "content": results})
                break  # end current turn, start fresh with compacted context

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
            output = tool_handler(**block.input) if tool_handler else f"Unknown: {block.name}"
            trigger_hooks("PostToolUse", block, output)
            if block.name == "todo_list":
                rounds_since_todo = 0
            print(output[:200])
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                }
            )
        else: # 防止和上下文压缩的results写入重复
            message.append({"role": "user", "content": results})
            continue



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
