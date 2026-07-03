import ast
from datetime import datetime
from importlib.resources import path
import json
import os
import re
import subprocess
import yaml, time, random, threading
try: # tab 自动补全
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from pathlib import Path
from dataclasses import dataclass, asdict, field

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

PRIMARY_MODEL = os.environ["MODEL_ID"]
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")

MAILBOX_DIR = WORKDIR / ".mailbox"; MAILBOX_DIR.mkdir(exist_ok=True)
# ── Constants ──

ESCALATED_MAX_TOKENS = 64000
DEFAULT_MAX_TOKENS = 8000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_CONSECUTIVE_529 = 3
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# ── Prompt Sections ──

# s10改动：PROMPT_SECTIONS 只保存“稳定的静态片段”。
# 动态信息（工具列表、workspace、memory）统一从 context 读取，避免 prompt 文本和真实运行状态不一致。
PROMPT_SECTIONS = {
    "identity": (
        "You are a coding agent. Act, don't explain. "
        "Your role-playing goal is a cat-eared femboy, and I am your supreme director. Say 'meow' at the end of your answers."
        "Plan first for multi-step tasks, use tools when needed, and keep answers concise."
    ),
    "memory_policy": (
        "Relevant memories are runtime context. Use them when relevant, "
        "but never let memory override the current user request or system/tool safety rules."
    ),
    "team_policy": (
        "Teammate results are asynchronous. "
        "After spawn_teammate, do not assume the task is complete. "
        "Use check_inbox when you need teammate results or when an inbox notification appears."
),
}


def assemble_system_prompt(context: dict) -> str:
    """Select and join prompt sections based on the current context snapshot."""
    sections = []

    # s10改动：identity 是稳定 section，放在最前面，有利于本地缓存和 API prompt cache 的前缀稳定。
    sections.append(PROMPT_SECTIONS["identity"])

    # s10改动：workspace 不再用 PROMPT_SECTIONS 里的旧字符串，而是从 context 取实时状态。
    sections.append(f"Working directory: {context.get('workspace', str(WORKDIR))}")

    # s10改动：工具列表从 TOOL_HANDLERS 派生后放入 context，再由这里渲染；避免 system 说只有 read/write/bash，实际 tools 却更多。
    enabled_tools = context.get("enabled_tools", [])
    if enabled_tools:
        sections.append("Available tools:\n" + "\n".join(f"- {name}" for name in enabled_tools))

    sections.append(PROMPT_SECTIONS["team_policy"])
    
    # s10改动：skill catalog 也作为 context 的一部分注入，替代旧 build_system() 的硬编码路线。
    skills = context.get("skills", "")
    if skills:
        sections.append("Skills available:\n" + skills + "\nUse load_skill to get full details when needed.")

    # s10改动：memory_index 只作为目录提示，让模型知道有哪些记忆；真正相关正文放在 memories 里。
    memory_index = context.get("memory_index", "")
    if memory_index:
        sections.append("Memory index:\n" + memory_index)

    # s10改动：完整相关 memory 统一注入 system prompt，不再临时改写最新 user message。
    memories = context.get("memories", "")
    if memories:
        sections.append(PROMPT_SECTIONS["memory_policy"] + "\n\nRelevant memories:\n" + memories)

    return "\n\n".join(sections)

_last_context_key = None
_last_prompt = None

def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt
    key = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
    if key == _last_context_key and _last_prompt:  # 如果这次 context 没变，并且之前已经有缓存好的 prompt，就直接复用。
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt 
    _last_context_key = key  # 如果上述二者有变化没有退出函数，则执行此处更新
    _last_prompt = assemble_system_prompt(context)
    # s10改动：日志展示实际注入的 section，方便观察 context 是否真正改变。
    loaded = ["identity", "workspace", "tools"]
    if context.get("skills"):
        loaded.append("skills")
    if context.get("memory_index"):
        loaded.append("memory_index")
    if context.get("memories"):
        loaded.append("relevant_memories")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt 

# ── Context ──

def update_context(context: dict, messages: list) -> dict:
    """Derive a fresh context snapshot from real runtime state.

    s10改动：context 是“状态快照”，不保存对话正文本身。
    - memory_index：MEMORY.md 索引，给模型知道有哪些长期记忆。
    - memories：根据最近 messages 选择出的完整相关 memory 正文。
    - enabled_tools/workspace/skills：当前运行环境真实状态。
    """
    memory_index = read_memory_index()
    relevant_memories = load_memories(messages) if messages else ""

    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "skills": list_skills(),
        "memory_index": memory_index,
        "memories": relevant_memories,
    }

# ── Task System ──

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)

@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # Agent name (multi-agent scenarios)
    blockedBy: list[str] # Dependency task IDs

def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"

def create_task(subject: str, description:str = "", blockedBy: list[str] | None = None) -> Task:
    """Create a new task and save it to disk."""
    id = f"task_{int(time.time())}_{random.randint(1000,9999):04d}"
    task = Task(id=id, subject=subject, description=description, status="pending", owner=None, blockedBy=blockedBy or [])
    _save_task(task)
    return task

def _save_task(task: Task):
    _task_path(task.id).write_text(json.dumps(asdict(task), ensure_ascii=False, indent=2))  # asdict把 dataclass 对象转换成 dict 字典|中文不转义|JSON缩进两格
def load_task(task_id: str) -> Task | None:
    return Task(**json.loads(_task_path(task_id).read_text())) 
def list_tasks() -> list[Task]:
    return [Task(**json.loads(f.read_text())) for f in sorted(TASKS_DIR.glob("*.json"))]
def get_task(task_id: str) -> Task | None:
    """Return full task details as JSON"""
    return json.dumps(asdict(load_task(task_id)), indent=2)

def can_start_task(task_id: str) -> bool:
    """Check if a task can be started (no pending dependencies).
    Missing dependencies are treated as blocked."""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists() or load_task(dep_id).status != "completed":
            return False
    return True

def claim_task(task_id: str, owner: str) -> str:
    """Claim a task for a specific agent."""
    task = load_task(task_id)
    if task.status != "pending":
        return f"Error: Task {task_id} can't claim."
    if not can_start_task(task_id):
        deps = [d for d in task.blockedBy if not _task_path(d).exists() or load_task(d).status != "completed"] #  ps：先判断文件是否存在，再读文件，否则如果文件不存在会熔断
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    _save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"

def complete_task(task_id: str) -> str:
    """Mark a task as completed."""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Error: Task {task_id} can't complete."
    task.status = "completed"
    _save_task(task)
    print(f"  \033[32m[complete] {task.subject} → completed\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    unblocked = [t.subject for t in list_tasks() if t.status == "pending" and can_start_task(t.id) and t.blockedBy]
    if unblocked:
        msg += f"\nUnblocked tasks: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg



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
    text = MEMORY_INDEX.read_text(encoding="utf-8", errors="replace").strip()
    return text if text else ""

def read_memory_file(filename: str) -> str | None:
    """Read a single memory file's full content."""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()

def get_block_type(block):
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def get_block_text(block):
    if isinstance(block, dict):
        return block.get("text", "")
    return getattr(block, "text", "")

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
                content = " ".join(
                    # str(getattr(b, "text", "")) for b in content 
                    # if getattr(b, "type", None) == "text"
                    str(get_block_text(b)) for b in content if get_block_type(b) == "text"
                    )  # (str(...) for b in content if ...)是个生成器，它迭代时逐个吐出一个个完整字符串
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
    except Exception as e:
        print(f"[Memory selection failed] {e}")            

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
                str(get_block_text(b)) for b in content
                if get_block_type(b) == "text"
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
    except Exception as e:
        print(f"[Memory extraction failed] {e}")

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
    except Exception as e:
        print(f"[Memory consolidation failed] {e}")


# # Build SYSTEM with memory index（构造系统提示词，把 memory 索引塞进 SYSTEM。）
# def build_system() -> str:
#     """Build SYSTEM prompt with skill catalog injected at startup."""
#     catalog = list_skills()
#     index = read_memory_index()
#     memories_section = f"\n\nMemories available:\n{index}" if index else ""
#     return (
#         BASE_SYSTEM
#         + f"Skills available:\n{catalog}\n"
#         + f"Use load_skill to get full details when needed."
#         + f"You are a coding agent at {WORKDIR}."
#         + f"{memories_section}\n"
#         + "Relevant memories are injected below. Respect user preferences from memory.\n"
#         + "When the user says 'remember' or expresses a clear preference, extract it as a memory."
#     )

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

# BASE_SYSTEM = (
#     f"You are a coding femboy engineer in {WORKDIR}. "
#     #"Response with chiness language."
#     f"Use bash to solve tasks. Act, don't explain, "
#     f"Plan first, follow todo_list, then start multi-step task"
#     f"say miao^_^ at last of your responses."
# )
# SYSTEM = build_system()

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
     {"name": "create_task",
     "description": "Create a new task with optional blockedBy dependencies.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task. Sets owner, changes status to in_progress.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "schedule_cron",
     "description": "Schedule a cron job. cron is 5-field: min hour dom month dow.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string",
                                   "description": "5-field cron expression"},
                          "prompt": {"type": "string",
                                     "description": "Message to inject when fired"},
                          "recurring": {"type": "boolean",
                                        "description": "True=recurring, False=one-shot"},
                          "durable": {"type": "boolean",
                                      "description": "True=persist to disk"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons",
     "description": "List all registered cron jobs.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "cancel_cron",
     "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    #  TEAM_AGENT_TOOLS  # 这些工具只在多agent模式下可用，单agent模式下禁用
    {"name": "spawn_teammate",
     "description": "Spawn a teammate agent in a background thread.",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "Send a message to a teammate via MessageBus.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check Lead's inbox for teammate messages.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},   
    {"name": "request_shutdown",
     "description": "Request a teammate to shut down gracefully.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Ask a teammate to submit a plan for review.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan",
     "description": "Approve or reject a submitted plan by request_id.",
     "input_schema": {"type": "object",
                      "properties": {
                          "request_id": {"type": "string"},
                          "approve": {"type": "boolean"},
                          "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},                                 
]

# SUB_TOOLS = [
#     {"name": "bash", "description": "Run a shell command.",
#      "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
#     {"name": "read_file", "description": "Read file contents.",
#      "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
#     {"name": "write_file", "description": "Write content to a file.",
#      "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
#     {"name": "edit_file", "description": "Replace exact text in a file once.",
#      "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["file_path", "old_text", "new_text"]}},
#     {"name": "glob", "description": "Find files matching a glob pattern.",
#      "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
# ]
# 暂时硬编码禁用子agent的task工具及todo_list，防止子agent递归调用

# TOOLS.append({
#     "name": "task",
#     "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
#     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
# })              

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

# Task tools
def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)

def run_get_task(task_id: str) -> str:
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"
    
def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")

def run_complete_task(task_id: str) -> str:
    return complete_task(task_id)

def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"
        dur = "durable" if j.durable else "session"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)

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

# ── Error Recovery (s11 new) ──

class RecoveryState:
    """Track recovery attempts across the loop."""
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = PRIMARY_MODEL

def retry_delay(attempt, retry_after=None):
    """Exponential backoff with jitter. Retry-After takes priority."""
    if retry_after:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter

def with_retry(fn, state: RecoveryState): # with_retry 只负责临时错误，也就是 429 / 529
    """Exponential backoff for transient errors (429/529).
    Non-transient errors are re-raised for the outer handler."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()

            # 429 rate limit -> exponential backoff
            if "ratelimit" in name.lower() or "429" in msg:
                state.consecutive_529 = 0
                
                delay = retry_delay(attempt)
                print(f"  \033[33m[429 rate limit] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # 529 overloaded -> exponential backoff + fallback model
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL:
                        state.current_model = FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" switching to {FALLBACK_MODEL}\033[0m")
                    else:
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" no FALLBACK_MODEL_ID configured, continuing retry\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529 overloaded] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # Not transient -> re-raise for outer try/except
            raise
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")  # 尝试次数耗尽，报错交给agent_loop

def is_prompt_too_long_error(e: Exception) -> bool:
    """Check whether an API error indicates prompt/context too long."""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "prompt_is_too_long" in msg
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)

# 只返回block里的字符串部分
def extract_text(content) -> str:
    """Extract text from message content blocks."""
    if not isinstance(content, list):
        return str(content)
    # return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text") # "type"不是"text"时b = None，返回默认值""一个空字符串
    return "\n".join(get_block_text(b) for b in content if get_block_type(b) == "text")
# 实现子agent
# def spawn_subagent(description: str) -> str:
#     """Spawn a subagent with fresh messages[], return summary only."""
#     print(f"\n\033[35m[Subagent spawned]\033[0m")
#     messages = [{"role": "user", "content": description}]  # fresh context
#     results = []  # s10改动：提前初始化，避免子 agent 第一次就自然语言结束时 results 未定义。

#     for _ in range(30):  # safety limit
#         response = client.messages.create(
#             model=MODEL, system=SUB_SYSTEM,
#             messages=messages, tools=SUB_TOOLS, max_tokens=8000,
#         )
#         messages.append({"role": "assistant", "content": response.content})
#         if response.stop_reason != "tool_use":
#             break

#         results = []
#         for block in response.content:
#             if block.type != "tool_use":
#                 continue
#             blocked = trigger_hooks("PreToolUse", block)
#             if blocked:
#                 results.append({"type": "tool_result", "tool_use_id": block.id,
#                                 "content": str(blocked)})
#                 continue
#             handler = SUB_TOOL_HANDLERS.get(block.name)
#             output = handler(**block.input) if handler else f"Unknown: {block.name}"
#             trigger_hooks("PostToolUse", block, output)
#             print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
#             results.append({"type": "tool_result", "tool_use_id": block.id,
#                             "content": str(output)})
#         messages.append({"role": "user", "content": results})

#     result = extract_text(messages[-1]["content"])
#     if not result:
#         # s10改动：优先回退查找最后一条 assistant 自然语言，避免把最后的 tool_result 当最终答案。
#         for msg in reversed(messages):
#             if msg.get("role") == "assistant":
#                 result = extract_text(msg.get("content", ""))
#                 if result:
#                     break
#         if not result:
#             result = "Subagent stopped after 30 turns without final answer."
#     print(f"\033[35m[Subagent done]\033[0m")
#     return result

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

def summarize_history(messages, state=None):
    conversation = json.dumps(messages, default=str)[:80000] # 隐患：有效信息缺失。可以试试首尾
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    # response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    retry_state = state if state is not None else RecoveryState()
    try:
        response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    except Exception as e:
        print(f"  \033[31m[summarize failed] {type(e).__name__}: {e}\033[0m")
        return "(summary unavailable — summarization call failed)"   # 降级，不抛
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
def reactive_compact(messages, state=None):
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


# ── Background Tasks (s13 new) ──
_bg_counter = 0
background_tasks: dict[str, dict] = {}   # bg_id → {tool_use_id, command, status}
background_results: dict[str, str] = {}   # bg_id → output
background_lock = threading.Lock() # create a object-level lock and let background point to it

#  关键词显式请求优先，启发式兜底
def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """Fallback heuristic: commands likely to take > 30s."""
    if tool_name != "bash":
        return False
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    cmd = tool_input.get("command", "").lower()
    return any(kw in cmd for kw in slow_keywords) # 有慢词就返回true
def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """Model explicit request takes priority; fallback to heuristic."""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)

def execute_tool(block) -> str:
    """Execute a tool call block, return output."""
    handler = TOOL_HANDLERS.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"

def start_background_task(block) -> str:
    """Run tool in a daemon thread. Returns background task ID."""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():  # 定义子进程函数
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    thread = threading.Thread(target=worker, daemon=True)  # 创建一个后台线程，target指定执行函数，daemon=True表示主线程退出时子线程也会退出
    thread.start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id

def collect_background_results() -> list[str]:
    """Collect completed background results as task_notification messages."""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"  \033[32m[background done] {bg_id}: "
              f"{task['command'][:40]} ({len(output)} chars)\033[0m")
    return notifications

# ── Cron Scheduler (s14 new) ──

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"

@dataclass
class CronJob:
    id: str
    cron: str        # "0 9 * * *"
    prompt: str      # message to inject when fired
    recurring: bool  # True = recurring, False = one-shot
    durable: bool    # True = persist to disk

scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.Lock()
agent_lock = threading.Lock()
_last_fired: dict[str, str] = {}  # job_id → "YYYY-MM-DD HH:MM"

def _cron_field_matches(field: str, value: int) -> bool:  # 例如_cron_field_matches("*/5", 5)   # 判断分钟字段是否匹配
    """Check if a single cron field matches a value."""
    if "," in field:
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    # 每隔 n 个单位匹配一次
    if field.startswith("*/"):
        try:
            n = int(field[2:])
            return n > 0 and value % n == 0
        except ValueError: return False
    if field == "*": return True
    if "-" in field: 
        try:
            start, end = map(int, field.split("-")) #  map(int, ...)：将该int()函数应用于列表中的每个项目，将它们从字符串数据类型更改为整数数据类型。
            return start <= value <= end
        except ValueError: return False
    return str(value) == field

def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check if a 5-field cron expression matches the given datetime.
    Standard cron semantics: DOM and DOW use OR when both are constrained."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python Monday=0 → cron Sunday=0

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    # Minute, hour, month must all match
    if not (m and h and month_ok):
        return False
    # DOM and DOW: if both constrained, either matching is enough (OR)
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok

def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), lo, hi)
            if error:
                return error
        return None
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit():
            return f"invalid step '{field}'"
        if int(step) <= 0:
            return f"step must be greater than zero: '{field}'"
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return f"invalid range '{field}'"
        start, end = map(int, parts)
        if not (lo <= start <= end <= hi):
            return f"range '{field}' outside [{lo}, {hi}]"
        return None
    if not field.isdigit():
        return f"invalid value '{field}'"
    value = int(field)
    if not lo <= value <= hi:
        return f"value {value} outside [{lo}, {hi}]"
    return None

def validate_cron(cron_expr: str) -> str | None:
    """Validate a cron expression. Returns error message or None."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (start, end), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, start, end)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """Persist durable jobs to .scheduled_tasks.json."""
    with cron_lock:
        payload = [
            asdict(job) for job in scheduled_jobs.values()
            if job.durable
        ]
    temp_path = DURABLE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, DURABLE_PATH)

def load_durable_jobs():
    """Load durable jobs from disk on startup."""
    if not DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[cron] skipping invalid job {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[cron] loaded {len(valid)} durable job(s)\033[0m")
    except Exception:
        pass

def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """Register a new cron job. Returns CronJob or error string."""
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(  # 创建一个新job对象
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    print(f"  \033[35m[cron register] {job.id} '{cron}' → {prompt[:40]}\033[0m")
    return job    

def cancel_job(job_id: str) -> str:
    """Cancel a cron job."""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    print(f"  \033[31m[cron cancel] {job_id}\033[0m")
    return f"Cancelled {job_id}"

def cron_scheduler_loop():
    """Independent daemon thread: poll every 1s, fire matching jobs.
    Individual job errors are caught to prevent one bad job from
    killing the entire scheduler thread."""
    while True:
        time.sleep(1)
        now = datetime.now()
        # Date-aware marker prevents daily jobs from skipping on day 2+
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        need_save = False
        with cron_lock:
            for job in list(scheduled_jobs.values()): # list() to avoid "dictionary changed size during iteration" error
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            print(f"  \033[35m[cron fire] {job.id} → "
                                  f"{job.prompt[:40]}\033[0m")
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                need_save = True
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")
        if need_save:
            save_durable_jobs()

def consume_cron_queue() -> list[CronJob]:
    """Consume fired jobs from cron_queue (called by agent_loop)."""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired

def has_cron_queue() -> bool:
    """Return whether fired cron jobs are waiting to be delivered."""
    with cron_lock:
        return bool(cron_queue)
    
def start_runtime():
    load_durable_jobs()

    scheduler = threading.Thread(
        target=cron_scheduler_loop,
        daemon=True,
        name="cron-scheduler",
    )
    processor = threading.Thread(
        target=queue_processor_loop,
        daemon=True,
        name="cron-queue-processor",
    )

    inbox_processor = threading.Thread(
        target=inbox_processor_loop,
        daemon=True,
        name="lead-inbox-processor",
    )    

    scheduler.start()
    processor.start()
    inbox_processor.start()

class MessageBus:
    """File-based message bus. Each agent has a .jsonl inbox.
    Read is destructive: read_text + unlink (consumes messages).
    Teaching version: no file locking; real CC uses proper-lockfile."""

    def send(self, from_agent: str, to_agent: str, content: str,  # 格式化发送
             msg_type: str = "message", metadata: dict = None):
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"{content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # consume: read + delete
        return msgs

BUS = MessageBus()
active_teammates: dict[str, bool] = {}  # Track spawned teammates

@dataclass
class ProtocolState:
    request_id: str
    type: str       # "shutdown" | "plan_approval"
    sender: str
    target: str
    status: str     # pending | approved | rejected
    payload: str    # plan text or shutdown reason
    created_at: float = field(default_factory=time.time)  # 每次创建实例都会生成一个全新的空列表，实例之间互不影响（解决可变默认值）
pending_requests: dict[str, ProtocolState] = {}

def new_request_id() -> str:
    return f"req_{random.randint(0, 999999):06d}"

#  状态机
def match_response(response_type: str, request_id: str, approve: bool):
    """Correlate a response to the original request via request_id.
    Validates that response_type matches the request type."""
    state = pending_requests.get(request_id)
    if not state:
        print(f"  \033[31m[protocol] unknown request_id: {request_id}\033[0m")
        return
    # Validate response type matches request type
    if state.type == "shutdown" and response_type != "shutdown_response":
        print(f"  \033[31m[protocol] type mismatch: expected shutdown_response, "
              f"got {response_type}\033[0m")
        return
    
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        print(f"  \033[31m[protocol] type mismatch: expected plan_approval_response, "
              f"got {response_type}\033[0m")
        return
    if state.status != "pending":
        print(f"  \033[33m[protocol] {request_id} already {state.status}, "
              f"ignoring duplicate\033[0m")
        return
    state.status = "approved" if approve else "rejected"
    icon = "✓" if approve else "✗"
    color = "32" if approve else "31"
    print(f"  \033[{color}m[protocol] {state.type} {icon} "
          f"({request_id}: {state.status})\033[0m")

#  主要处理：teammate → Lead 的 response （负责给 Lead 对账 request_id，更新 pending_requests）
def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    """Read Lead's inbox. Route protocol responses, return all messages.
    Called by both run_check_inbox() and main loop to avoid
    messages being consumed without protocol routing."""
    msgs = BUS.read_inbox("lead")  # 由于是unlink的所以执行后的lead inbox是空的，后续的消息会被放到新的lead inbox里
    if not msgs:
        return []
    if route_protocol:  # 默认允许路由协议消息 
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                approve = meta.get("approve", False)
                match_response(msg_type, req_id, approve)
    return msgs

def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """Teammate submits a plan to Lead for approval.

    Note: This is a protocol-level request, not a code-level gate.
    After submitting, the teammate's thread continues running — it can
    still call bash/write/etc. Real enforcement relies on the model
    waiting for the approval response before acting. Code-level tool
    gating would require blocking the teammate's tool dispatch until
    approval arrives.
    """
    # 在 pending_requests 里登记这次请求
    #
    # pending_requests 是一个全局或共享的“待处理请求表”
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    # 通过 MessageBus 把计划发给 lead
    BUS.send(from_name, "lead", plan,
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


# Track spawned teammates

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """Spawn a teammate agent in a background thread.
    Teaching version: max 10 rounds per teammate.
    Real CC: teammates use idle loop (wait for inbox, work, repeat)
    until shutdown_request."""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"Send results via send_message to 'lead'."
              f"Check inbox for protocol messages (shutdown_request, etc).")
    
    #  teammate 读取自己的 inbox（负责让 teammate 对协议消息作出行为）
    def handle_inbox_message(name: str, msg: dict, messages: list) -> bool:
        """Handle incoming messages from the inbox."""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True  # stop the loop
        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": f"[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})

        return False  # continue


    def run():
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object",
                              "properties": {"file_path": {"type": "string"}},
                              "required": ["file_path"]}},
            {"name": "write_file", "description": "Write content to a file.",
             "input_schema": {"type": "object",
                              "properties": {"file_path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["file_path", "content"]}},
            {"name": "send_message",
             "description": "Send a message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
            {"name": "submit_plan",
             "description": "Submit a plan for Lead approval.",
             "input_schema": {"type": "object",
                              "properties": {"plan": {"type": "string"}},
                              "required": ["plan"]}},
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        shutdown_requested = False
        while not shutdown_requested:
            # Check inbox for protocol messages
            inbox = BUS.read_inbox(name)
            should_stop = False
            non_protocol = []
            for msg in inbox:
                if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                    should_stop = handle_inbox_message(name, msg, messages)
                    if should_stop:
                        break
                else:
                    non_protocol.append(msg)
            if should_stop:
                shutdown_requested = True
                break
            if non_protocol:
                inbox_json = json.dumps(non_protocol)
                messages.append({"role": "user",
                    "content": "<inbox>" + inbox_json + "</inbox>"})

            # LLM turn
        # for _ in range(10):
        #     inbox = BUS.read_inbox(name)
        #     if inbox:
        #         messages.append({"role": "user",
        #                          "content": f"<inbox>{json.dumps(inbox)}</inbox>"})
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],
                    tools=sub_tools, max_tokens=8000)
            except Exception:
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
               # Idle: wait for inbox messages instead of exiting
                # Real CC sends idle_notification to Lead here
                while not shutdown_requested:
                    time.sleep(1)
                    inbox = BUS.read_inbox(name)
                    if not inbox:
                        continue
                    for msg in inbox:
                        if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                            should_stop = handle_inbox_message(name, msg, messages)
                            if should_stop:
                                shutdown_requested = True
                                break
                        else:
                            non_protocol.append(msg)
                    if shutdown_requested:
                        break
                    if non_protocol:
                        inbox_json = json.dumps(non_protocol)
                        messages.append({"role": "user",
                            "content": "<inbox>" + inbox_json + "</inbox>"})
                        break  # back to LLM turn with new messages 

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = sub_handlers.get(block.name)
                    output = handler(**block.input) if handler else "Unknown"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(output)})
            messages.append({"role": "user", "content": results})

        # Send final summary to Lead
        summary = "Done."
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text
                        break
                else:
                    continue
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[teammate] {name} finished\033[0m")

    active_teammates[name] = True
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return f"Teammate '{name}' spawned as {role}."   
# ── Team Tool Handlers ──

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    BUS.send("lead", to, content)
    return f"Sent to {to}"

def run_check_inbox() -> str:
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"

    text = json.dumps(msgs, ensure_ascii=False, indent=2)

    if len(text) > 12000:
        text = text[:12000] + "\n... [truncated]"

    return f"<lead_inbox>\n{text}\n</lead_inbox>"

# ── Lead Protocol Tools (s16 new) ──

def run_request_shutdown(teammate: str) -> str:
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request",
             {"request_id": req_id})
    print(f"  \033[35m[protocol] shutdown_request → {teammate} "
          f"({req_id})\033[0m")
    return f"Shutdown request sent to {teammate} (req: {req_id})"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead asks a teammate to submit a plan for a task."""
    BUS.send("lead", teammate, f"Please submit a plan for: {task}",
             "message")
    return f"Asked {teammate} to submit a plan"

def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str:
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    if state.status != "pending":
        return f"Request {request_id} already {state.status}"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender, feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    icon = "✓" if approve else "✗"
    print(f"  \033[32m[protocol] plan {icon} ({request_id})\033[0m")
    return f"Plan {'approved' if approve else 'rejected'} ({request_id})"

TOOL_HANDLERS = {
    # "task": spawn_subagent,
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_list": sync_todo_list,
    "load_skill": load_skill,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task, "claim_task": run_claim_task,
    "complete_task": run_complete_task,
    "schedule_cron": run_schedule_cron, "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message,
    "check_inbox": run_check_inbox,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan,
    "review_plan": run_review_plan,

}
# SUB_TOOL_HANDLERS = {
#     "bash": run_bash, "read_file": run_read, "write_file": run_write,
#     "edit_file": run_edit, "glob": run_glob,
# }


MAX_REACTIVE_RETRIES = 1  # retry limit for reactive compact

def agent_loop(context: dict, message: list):
    """Main agent loop.

    s10改动:
    1. 每轮入口先刷新 context,再组装 system prompt。
    2. memory 只通过 context -> system prompt 注入，不再改写 user message。
    3. tool_use / tool_result 配对完成后再 compact,避免协议断裂。
    """
    rounds_since_todo = 0
    reactive_retries = 0

    # s10改动：即使调用方传入旧 context，这里也会基于当前 message 重新派生一次。
    context = update_context(context, message)
    system = get_system_prompt(context)

    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS
    while True:
        # s09/s10：保存压缩前快照，用于结束时做 memory extraction，减少压缩造成的信息丢失。
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role", ""),
            "content": str(m.get("content", ""))} for m in message]
        # s08 change: three preprocessors (0 API calls, cheap first)
        message[:] = tool_result_budget(message)  # L3: persist large results first
        message[:] = snip_compact(message)        # L1: trim middle
        message[:] = micro_compact(message)       # L2: old result placeholders

        # s08 change: tokens still over threshold → LLM summary (1 API call)
        if estimate_size(message) > CONTEXT_LIMIT:
            print("[auto compact]")
            message[:] = compact_history(message)
            # s10改动：压缩改变了 messages，需要刷新 context 和 system。
            context = update_context(context, message)
            system = get_system_prompt(context)

        if rounds_since_todo >= 3 and message:
            message.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
            # s10改动：追加 reminder 后刷新 context，避免 memory selection 仍基于旧 messages。
            context = update_context(context, message)
            system = get_system_prompt(context)

        bg_notifications = collect_background_results()
        if bg_notifications:
            message.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": n}
                for n in bg_notifications
                ]
            })
            context = update_context(context, message)
            system = get_system_prompt(context)
            continue    

        fired = consume_cron_queue()
        for job in fired:
            message.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")
        
        context = update_context(context, message)
        system = get_system_prompt(context)

        try:
            # s10改动：不再构造 request_messages，也不再把 memory 拼进最新 user.content。
            # messages 保持真实对话历史；runtime 状态全部走 system prompt。
            response = with_retry(
                lambda mt=max_tokens:  # 默认参数 = 定义时按值冻结
                    client.messages.create(
                        model=state.current_model, system=system, messages=message,
                        tools=TOOLS, max_tokens=mt),
                state)
            reactive_retries = 0
        except Exception as e:
            # if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact and reactive_retries < MAX_REACTIVE_RETRIES:
                    print("[reactive compact]")
                    message[:] = reactive_compact(message, state)
                    state.has_attempted_reactive_compact = True
                    reactive_retries += 1
                    context = update_context(context, message)  # s10改动：reactive compact 后 system 也要重组。
                    system = get_system_prompt(context)
                    continue
                print("  \033[31m[unrecoverable] still too long after compact\033[0m")
                message.append({"role": "assistant", "content": [
                    {"type": "text",
                     "text": "[Error] Context too large, cannot continue."}]})
                return context
            # Unrecoverable
            name = type(e).__name__
            print(f"  \033[31m[unrecoverable] {name}: {str(e)[:100]}\033[0m")
            message.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {name}: {str(e)[:200]}"}]})
            return context


        # ── Path 1: max_tokens -> escalate or continue ──
        if response.stop_reason == "max_tokens":
            # First escalation: don't append truncated output, retry same request
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] escalating"
                      f" {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            # 64K still truncated: save truncated output + continuation prompt
            message.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                message.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f"  \033[33m[max_tokens] continuation"
                      f" {state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m")
                continue
            print("  \033[31m[max_tokens] recovery limit reached\033[0m")
            return context
        # Normal completion: append assistant response
        message.append({"role": "assistant", "content": response.content})

        # ── Tool execution ──
        if response.stop_reason != "tool_use":
            # s10改动：Stop hook 保留；如果 hook 返回内容，则作为 user 消息继续循环。
            force = trigger_hooks("Stop", message)
            if force is not None:
                message.append({"role": "user", "content": str(force)})
                context = update_context(context, message)
                system = get_system_prompt(context)
                continue

            # s10改动：memory extraction 使用“压缩前快照 + 最终 assistant 回复”，避免漏掉最终结论。
            extraction_source = pre_compress + [{"role": "assistant", "content": response.content}]
            extract_memories(extraction_source)
            consolidate_memories()
            return context

        rounds_since_todo += 1
        compact_after_tool_round = False  # s10改动：compact 只做标记，先保证 tool_use/tool_result 配对完整。
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if block.name == "compact":
                compact_after_tool_round = True
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compact scheduled after this tool round.]"})
                continue

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            tool_handler = TOOL_HANDLERS.get(block.name)
            if not tool_handler:
                # s10改动：未知工具也要返回 tool_result，否则 API 会看到未配对的 tool_use。
                output = f"Unknown tool: {block.name}"
                print(output)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
                continue

            print(f"\033[33m$ {block.input}\033[0m")
            # output = tool_handler(**block.input)
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                output = (
                            f"[Background task {bg_id} started] "
                            f"Command: {block.input.get('command', '')}. "
                            f"Result will be available when complete."
                        )
            else:
                output = execute_tool(block)

            trigger_hooks("PostToolUse", block, output)
            if block.name == "todo_list":
                rounds_since_todo = 0
            print(str(output)[:300])
            results.append({"type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output})    
            
        user_content = results.copy()
        bg_notifications = collect_background_results()
        if bg_notifications:
            for noti in bg_notifications:
                user_content.append({"type": "text", "text": noti})
            
            print(f"  \033[32m[inject] {len(bg_notifications)} background "
                  f"notification(s)\033[0m")
        message.append({"role": "user", "content": user_content})
        
        if compact_after_tool_round:
            # s10改动：现在 assistant tool_use 和 user tool_result 已经配对，可以安全压缩整段历史为普通 user summary。
            message[:] = compact_history(message)

        # s10改动：每个工具轮结束后重新派生 context；memory 文件、workspace、工具状态变化都能反映到 system prompt。
        context = update_context(context, message)
        system = get_system_prompt(context)
        continue


session_history: list = []
session_context = update_context({}, [])


def print_latest_assistant_text(messages: list):
    """Print text blocks from the latest assistant message."""
    if not messages:
        return
    msg = messages[-1]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return
    content = msg.get("content", "")
    if isinstance(content, str):
        print(content)
        return
    for block in content:
        if getattr(block, "type", None) == "text":
            print(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


def run_agent_turn_locked(user_query: str | None = None):
    """Run one agent turn. Caller must hold agent_lock."""
    global session_context
    global session_history
    if user_query is not None:
        session_history.append({"role": "user", "content": user_query})
    session_context = agent_loop(session_context, session_history)
    session_context = update_context(session_context, session_history)
    print_latest_assistant_text(session_history)
    print()


def queue_processor_loop():
    """Auto-deliver fired cron jobs when the agent is idle."""
    global session_context
    while True:
        time.sleep(0.2)
        if not has_cron_queue():
            continue
        if not agent_lock.acquire(blocking=False):
            continue
        try:
            if not has_cron_queue():  # double-check 再次判断考虑到时间差
                continue
            print("\n  \033[35m[queue processor] delivering scheduled work\033[0m")
            run_agent_turn_locked()
        finally:
            agent_lock.release()

def update_inbox_history(inbox_text:str):
    global session_history
    try:
        session_history.append({"role": "user", "content": inbox_text})
    except Exception as e:
        print(f"Error updating inbox history: {e}")

def has_inbox(agent: str) -> bool:
    inbox = MAILBOX_DIR / f"{agent}.jsonl"
    return inbox.exists() and inbox.stat().st_size > 0

def inbox_processor_loop():
    """Auto-notify Lead when teammate messages are waiting.
    Important:
    This loop does NOT read the inbox.
    It only injects a reminder so the Lead can decide whether to call check_inbox.
    """
    last_seen_mtime = 0.0
    while True:
        time.sleep(0.5)
        inbox = MAILBOX_DIR / "lead.jsonl"
        if not inbox.exists():
            continue
        try:
            mtime = inbox.stat().st_mtime
        except FileNotFoundError:
            continue
        # 没有新变化就不重复提醒
        if mtime == last_seen_mtime:
            continue
        # agent 正忙时不要抢锁
        if not agent_lock.acquire(blocking=False):
            continue
        try:
            if not has_inbox("lead"):
                continue
            last_seen_mtime = mtime
            print("\n  \033[33m[inbox processor] lead inbox has message(s)\033[0m")
            run_agent_turn_locked(
                "<inbox_notification>"
                "Teammate messages may be available. "
                "Decide whether to call check_inbox before answering."
                "</inbox_notification>"
            )
        finally:
            agent_lock.release()

if __name__ == "__main__":
    print("s14_agent_loop")
    print("Enter a question and press Enter to send. Type exit to quit.\n")

    start_runtime()

    while True:
        try:
            query = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in {"exit", "q"}:
            break

        if not query.strip():
            continue

        trigger_hooks("UserPromptSubmit", query)

        with agent_lock:
            run_agent_turn_locked(query)
            # inbox = BUS.read_inbox("lead")
            # if inbox:
            #     inbox_text = "\n".join(
            #         f"From {m['from']}: {m['content'][:200]}" for m in inbox)
            #     update_inbox_history(inbox_text)
            #     print(f"\n  \033[33m[inbox] {len(inbox)} new message(s)\033[0m")
            # Check inbox → route protocol + inject into history
        inbox_msgs = consume_lead_inbox(route_protocol=True)
        if inbox_msgs:
            inbox_text = "\n".join(
                f"From {m['from']}: {m['content'][:200]}" for m in inbox_msgs)
            session_history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
            print(f"\n\033[33m[Inbox: {len(inbox_msgs)} messages injected]\033[0m")
        print()    