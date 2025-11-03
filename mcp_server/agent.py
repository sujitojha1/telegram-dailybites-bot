# agent.py

import asyncio
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from core.loop import AgentLoop
from core.session import MultiMCP

console = Console()


def format_final_answer(raw: str) -> str:
    """Normalize FINAL_ANSWER payload for pretty console output."""
    answer = raw.replace("FINAL_ANSWER:", "").strip()
    if answer.startswith("[") and answer.endswith("]"):
        answer = answer[1:-1].strip()
    answer = answer.replace("\\n", "\n").strip()

    lines = answer.splitlines()
    if len(lines) > 1 and lines[0].strip() and lines[1].strip().startswith("1."):
        lines.insert(1, "")  # add spacing between lead-in sentence and list
        answer = "\n".join(lines)

    return answer or "No answer returned."

def log(stage: str, msg: str):
    """Simple timestamped console logger."""
    import datetime
    now = datetime.datetime.now().strftime("%H:%M:%S")
    console.print(f"[{now}] [{stage}] {msg}")


async def main():
    console.print("🧠 Telegram Agent Ready", style="bold green")
    user_input = "find top 3 latest Hacker News article - https://news.ycombinator.com/ with links and for the content in to email with html tags before sending email"
    # user_input = input("🧑 What do you want to solve today? → ")
    console.print(f"🧑 What do you want to solve today? → {user_input}")

    # Load MCP server configs from profiles.yaml
    with open("config/profiles.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers = profile.get("mcp_servers", [])

    multi_mcp = MultiMCP(server_configs=mcp_servers)
    #console.print("\nAgent before initialize", style="dim")
    await multi_mcp.initialize()

    agent = AgentLoop(
        user_input=user_input,
        dispatcher=multi_mcp  # now uses dynamic MultiMCP
    )

    try:
        final_response = await agent.run()
        answer = format_final_answer(final_response)
        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="💡 Final Answer",
                border_style="cyan",
                expand=False,
            )
        )

    except Exception as e:
        log("fatal", f"Agent failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
