from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## absolute path to the server
                "transport":"stdio",
            },
            "weather": {
                "command":"python",
                "args":["weather.py"], ## absolute path to the server
                "transport":"stdio",
            }
        }
    )

    import os
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    tools = await client.get_tools()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022")
    agent = create_react_agent(
        model, tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in LA?"}]}
    )

    print("math response:", math_response['messages'][-1].content)

asyncio.run(main())

