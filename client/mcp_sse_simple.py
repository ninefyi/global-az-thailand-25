import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from mcp_use import set_debug, MCPClient, MCPAgent

set_debug(2)

async def main():
    """Run the example using a configuration file."""
    # Load environment variables
    load_dotenv()

    config = {
        "mcpServers": {
            "http": {
                "url": "http://localhost:5008/sse"
            }
        }
    }

    # Create MCPClient from config file
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = AzureChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "List all Azure subscriptions",
        max_steps=30,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())