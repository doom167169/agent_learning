import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import datetime
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@tool
def get_time() -> str:
    """Get current time"""
    return datetime.date.today().isoformat()


async def main():
    client = MultiServerMCPClient(
        {"travel_server": {"transport": "http", "url": "https://mcp.kiwi.com"}}
    )
    tools = await client.get_tools()
    print("loaded tools:", [t.name for t in tools])

    # Fake model: first respond with a tool_call to search-flight, then final answer.
    fake = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search-flight",
                             "args": {"origin": "PRG", "destination": "PAR",
                                      "dateOfDeparture": "2026-08-10",
                                      "limit": 1},
                             "id": "call_1", "type": "tool_call"}],
            ),
            AIMessage(content="done"),
        ]
    )

    agent = create_agent(
        model=fake,
        tools=[*tools, get_time],
        system_prompt="you are a test agent",
    )
    print("agent compiled")

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="search a flight")]}
        )
        print("OK result messages:", len(result["messages"]))
        for m in result["messages"]:
            print(" -", type(m).__name__, repr(getattr(m, "content", ""))[:120])
    except Exception as e:
        import traceback
        print("RAISED:", type(e).__name__, e)
        traceback.print_exc()


asyncio.run(main())
