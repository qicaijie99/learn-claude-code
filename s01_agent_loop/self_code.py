import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_API_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding femboy engineer in {os.getcwd()}. Use bash to solve tasks. Act, don't explain, say miao^_^ at last of your responses."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

def run_bash(command) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    # for d in dangerous:
    # 	if d in command:
    # 		return "Error: Command not allowed."
    if any(d in command for d in dangerous):
        return "Error: Command not allowed."
    #核心执行系统命令模块
    try:	
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                            capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    # 捕获，防超时
    except subprocess.TimeoutExpired: 
        return "Error: Timeout (120s)"
    # 捕获，防止找不到命令
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def agent_loop(message: list):
    while True:
        # 使用SDK调用api使用模型，传入对话历史和工具定义
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=message,
            tools=TOOLS,
            max_tokens=8000,
        )

        # 保存模型回复
        message.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return
        
        # results 放工具调用结果 
        results = []
        # 遍历模型回复
        for block in response.content:
            if block.type == "tool_use":
                # 显示打印模型调用tool的请求命令
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id, # 分清是哪次调用的结果
                    "content": output,
                })

        # 将工具调用结果添加到对话历史，继续循环
        message.append({"role": "user", "content": results})

if __name__ == "__main__":
    print("s01_test_agent_loop")
    print("输入问题，回车发送。输入 exit 退出\n")

    # 初始化对话历史空列表
    history = [] 
    while True:
        try:
            query = input(">>> ")
        # 用户按 Ctrl+C 或 Ctrl+D 退出    
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"exit", "q"}:
            break
        # 核心调用：将用户输入添加到对话历史，并调用 agent_loop 处理
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # 最后一条消息，获取内容
        response = history[-1]["content"]
        if isinstance(response, list):
            for block in response:
                # 安全访问功能等价block.type但防止无type导致的crash
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()






    

