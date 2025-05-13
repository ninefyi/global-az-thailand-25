import asyncio
import os
import json
from typing import List
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import AzureChatOpenAI
from mcp_use import MCPAgent, MCPClient
from mcp import ClientSession
from mcp.client.sse import sse_client

# Load environment variables
load_dotenv()

# Initialize event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Global agent and client variables
mcp_client = None
agent = None

async def initialize_agent(server_url):
    """Initialize the MCP client and agent."""
    global mcp_client, agent
    
    config = {
        "mcpServers": {
            "http": {
                "url": server_url
            }
        }
    }
    
    try:
        # Initialize the MCP client and LLM
        mcp_client = MCPClient.from_dict(config)
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=0.7
        )
        agent = MCPAgent(llm=llm, client=mcp_client, max_steps=30, verbose=True)

        print(f"Connecting to MCP server at {server_url}...")
        
        # Extract the base URL from the SSE endpoint
        base_url = server_url.rsplit('/', 1)[0] if '/sse' in server_url else server_url
        sse_endpoint = server_url if '/sse' in server_url else f"{server_url.rstrip('/')}/sse"
        
        try:
            async with sse_client(sse_endpoint) as streams:
                async with ClientSession(*streams) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # List available tools to verify connection
                    print("Initialized SSE client...")
                    print("Listing tools...")
                    response = await session.list_tools()
                    tools = response.tools
                    tool_names = [tool.name for tool in tools]
                    print(f"Connected to MCP server. Available tools:", tool_names)
                    
                    # Return success message with available tools
                    return "Connected to MCP server. Available tools:", tool_names
        except Exception as inner_e:
            raise ConnectionError(f"Failed to establish session with MCP server: {str(inner_e)}")
            
    except Exception as e:
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return f"Error connecting to MCP server: {str(e)}"

def connect(server_url):
    """Connect to the MCP server and initialize the agent."""
    return asyncio.run(initialize_agent(server_url))

async def process_message_async(message, history):
    """Process a user message using the MCP agent."""
    global agent
    
    if not agent:
        return history + [(message, "Please connect to an MCP server first.")], ""
    
    try:
        # Run the query through the agent
        result = await agent.run(message, max_steps=30)
        return history + [(message, result)], ""
    except Exception as e:
        return history + [(message, f"Error: {str(e)}")], ""

def process_message(message, history):
    """Synchronous wrapper for process_message_async."""
    result, empty_text = asyncio.run(process_message_async(message, history))
    return result, empty_text

def gradio_interface():
    """Create a Gradio interface for the MCP agent."""
    with gr.Blocks(title="MCP Azure Client") as demo:
        gr.Markdown("# MCP Azure Assistant")
        gr.Markdown("Connect to your MCP Azure server and chat with the assistant")
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                server_path = gr.Textbox(
                    label="Server-Sent Events (SSE) URL",
                    placeholder="Enter path to sse url",
                    value="http://localhost:5008/sse"
                )
            with gr.Column(scale=1):
                connect_btn = gr.Button("Connect")
        
        status = gr.Textbox(label="Connection Status", interactive=False)
        
        chatbot = gr.Chatbot(
            value=[], 
            height=200,
            show_copy_button=True
        )
        
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about Azure (e.g., List my resource groups)",
                scale=4
            )
            clear_btn = gr.Button("Clear Chat", scale=1)
        
        # Set up event handlers
        connect_btn.click(connect, inputs=server_path, outputs=status)
        msg.submit(process_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
        
        # Add example prompts
        gr.Examples(
            examples=[
                "List my Azure subscriptions",
                "Show my virtual machines",
                "Tell me about my resource groups",
            ],
            inputs=msg
        )
        
    return demo

if __name__ == "__main__":
    # Environment variable checks
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"Warning: {var} not found in environment. Please set it in your .env file.")
    
    interface = gradio_interface()
    interface.launch(debug=True, share=True)