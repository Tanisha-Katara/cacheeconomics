
import asyncio, os, sys
from browser_use import Agent
from browser_use.llm.anthropic.chat import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5",
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    base_url=os.environ["CAPTURE_BASE_URL"])
TASK = ("Go to https://en.wikipedia.org/wiki/Special:RecentChanges and report "
        "the title of the most recent change listed.")
async def main():
    a = Agent(task=TASK, llm=llm, max_actions_per_step=2)
    try:
        await asyncio.wait_for(a.run(max_steps=4), timeout=240)
    except Exception as e:
        print("cycle ended:", type(e).__name__, str(e)[:120], file=sys.stderr)
asyncio.run(main())
