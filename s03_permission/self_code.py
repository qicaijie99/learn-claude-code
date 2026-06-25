import os
import subprocess

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_API_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()
SYSTEM = (
    f"You are a coding femboy engineer in {os.getcwd()}. "
    "Use bash to solve tasks. Act, don't explain, "
    "say miao^_^ at last of your responses."
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
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
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
]


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
        lines = safe_path(file_path).read_text().splitlines() # 把读到的字符串按行分割成list = ["line1", "line2", ...]
        if limit and limit < len(lines):
            lines = lines[:limit]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(file_path: str, content: str) -> str:
    try:
        file_path = safe_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(file_path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(file_path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {file_path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
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

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# 第一层硬编码防御
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/sda", "mkfs", "dd if=", "dd of="]
def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Error: Command '{command}' is not allowed"
    return None

# 第二层权限检查 (路径外写入/编辑未赋权及危险指令)
PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    {"tools": ["bash"],
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]

def check_permission(tool_name: str, args: str) -> str | None: # 从字典中取出工具名和参数（键值对）
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"]:
            if rule["check"](args):
                return rule["message"]
    return None

# 第三层等待用户输入
def ask_user_permission(tool_name: str, args: dict, reason: str) -> bool:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"

def check_permission_pipeline(block) -> bool:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
    reason = check_permission(block.name, block.input)
    if reason:
        decision = ask_user_permission(block.name, block.input, reason)
        if decision == "deny":
            return False
    return True        

def agent_loop(message: list):
    while True:
        # 最初交给LLM的消息是用户输入的query以及下列基本信息，之后每次循环都会把上一次的response作为新的message传给LLM
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=message,
            tools=TOOLS,
            max_tokens=8000,
        )

        message.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m> {block.name}\033[0m")
                if not check_permission_pipeline(block):
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Permission denied.",
                        }
                    )
                    continue

                tool_handler = TOOL_HANDLERS.get(block.name)
                if not tool_handler:
                    print(f"Error: Unknown tool '{block.name}'")
                    continue
                print(f"\033[33m$ {block.input}\033[0m")
                output = tool_handler(**block.input)
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
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response = history[-1]["content"]
        if isinstance(response, list):
            for block in response:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
