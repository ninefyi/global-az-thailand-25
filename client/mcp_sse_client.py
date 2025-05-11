import asyncio
import os
import json
from typing import List, Dict, Any, Union
from contextlib import AsyncExitStack

import gradio as gr
from gradio.components.chatbot import ChatMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class MCPClientWrapper:
    def __init__(self):
        self.session = None
        self.exit_stack = None
        
        # Replace Anthropic with Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Azure OpenAI deployment name
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        self.tools = []
    
    def connect(self, server_path: str) -> str:
        return loop.run_until_complete(self._connect(server_path))


    async def _connect(self, server_path: str) -> str:
        if self.exit_stack:
            await self.exit_stack.aclose()
        
        self.exit_stack = AsyncExitStack()
        
        sse_transport = await self.exit_stack.enter_async_context(sse_client(server_path))
        self.read, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        
        response = await self.session.list_tools()
        self.tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema if tool.inputSchema else {}
            }
        } for tool in response.tools]

        # Validate tool schemas
        for tool in self.tools:
            parameters = tool["function"]["parameters"]
            if not isinstance(parameters, dict) or "type" not in parameters or parameters["type"] != "object":
                tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

        tool_names = [tool["function"]["name"] for tool in self.tools]
        return f"Connected to MCP server. Available tools: {', '.join(tool_names)}"
    
    def process_message(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]) -> tuple:
        if not self.session:
            return history + [
                {"role": "user", "content": message}, 
                {"role": "assistant", "content": "Please connect to an MCP server first."}
            ], gr.Textbox(value="")
        
        new_messages = loop.run_until_complete(self._process_query(message, history))
        return history + [{"role": "user", "content": message}] + new_messages, gr.Textbox(value="")
    
    async def _process_query(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):
        openai_messages = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")
            
            if role in ["user", "assistant", "system"]:
                openai_messages.append({"role": role, "content": content})
        
        openai_messages.append({"role": "user", "content": message})
        
        # Convert to Azure OpenAI call
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=openai_messages,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=1000
        )

        result_messages = []
        
        # Handle assistant response
        assistant_message = response.choices[0].message
        
        if not assistant_message.tool_calls:
            # Simple text response
            result_messages.append({
                "role": "assistant", 
                "content": assistant_message.content
            })
        else:
            # Tool calls processing
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                result_messages.append({
                    "role": "assistant",
                    "content": f"I'll use the {tool_name} tool to help answer your question.",
                    "metadata": {
                        "title": f"Using tool: {tool_name}",
                        "log": f"Parameters: {json.dumps(tool_args, ensure_ascii=True)}",
                        "status": "pending",
                        "id": f"tool_call_{tool_name}"
                    }
                })
                
                result_messages.append({
                    "role": "assistant",
                    "content": "```json\n" + json.dumps(tool_args, indent=2, ensure_ascii=True) + "\n```",
                    "metadata": {
                        "parent_id": f"tool_call_{tool_name}",
                        "id": f"params_{tool_name}",
                        "title": "Tool Parameters"
                    }
                })
                
                result = await self.session.call_tool(tool_name, tool_args)
                
                if result_messages and "metadata" in result_messages[-2]:
                    result_messages[-2]["metadata"]["status"] = "done"
                
                result_messages.append({
                    "role": "assistant",
                    "content": "Here are the results from the tool:",
                    "metadata": {
                        "title": f"Tool Result for {tool_name}",
                        "status": "done",
                        "id": f"result_{tool_name}"
                    }
                })
                
                result_content = result.content
                if isinstance(result_content, list):
                    result_content = "\n".join(str(item) for item in result_content)
                
                try:
                    result_json = json.loads(result_content)
                    if isinstance(result_json, dict) and "type" in result_json:
                        if result_json["type"] == "image" and "url" in result_json:
                            result_messages.append({
                                "role": "assistant",
                                "content": {"path": result_json["url"], "alt_text": result_json.get("message", "Generated image")},
                                "metadata": {
                                    "parent_id": f"result_{tool_name}",
                                    "id": f"image_{tool_name}",
                                    "title": "Generated Image"
                                }
                            })
                        else:
                            result_messages.append({
                                "role": "assistant",
                                "content": "```\n" + result_content + "\n```",
                                "metadata": {
                                    "parent_id": f"result_{tool_name}",
                                    "id": f"raw_result_{tool_name}",
                                    "title": "Raw Output"
                                }
                            })
                except:
                    result_messages.append({
                        "role": "assistant",
                        "content": "```\n" + result_content + "\n```",
                        "metadata": {
                            "parent_id": f"result_{tool_name}",
                            "id": f"raw_result_{tool_name}",
                            "title": "Raw Output"
                        }
                    })
                
                # Add tool response to history and get assistant's follow-up
                openai_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    }]
                })
                
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })
                
                # Get the assistant's response after tool call
                next_response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=openai_messages,
                    max_tokens=1000
                )
                
                if next_response.choices[0].message.content:
                    result_messages.append({
                        "role": "assistant",
                        "content": next_response.choices[0].message.content
                    })

        return result_messages

client = MCPClientWrapper()

def gradio_interface():
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
            height=500,
            type="messages",
            show_copy_button=True,
            avatar_images=("👤", "🤖")
        )
        
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about Azure (e.g., List my resource groups)",
                scale=4
            )
            clear_btn = gr.Button("Clear Chat", scale=1)
        
        connect_btn.click(client.connect, inputs=server_path, outputs=status)
        msg.submit(client.process_message, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
        
    return demo

if __name__ == "__main__":
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Warning: AZURE_OPENAI_API_KEY not found in environment. Please set it in your .env file.")
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Warning: AZURE_OPENAI_ENDPOINT not found in environment. Please set it in your .env file.")
    if not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        print("Warning: AZURE_OPENAI_DEPLOYMENT not found in environment. Please set it in your .env file.")
    
    interface = gradio_interface()
    interface.launch(debug=True, share=True)