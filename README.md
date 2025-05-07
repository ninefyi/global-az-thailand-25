# Global Azure Thailand 2025
- Date: Sat, 10 May 2025
- Topic: Introduction to the Azure MCP Server
- Event type: Online
----
## Instruction
1. `sudo apt-get update && sudo apt-get install -y azure-cli`
3. `az login`
4. Click URL link and copy code
5. Pasted code to the URL link.
6. create `.env`
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_api_key_endpoint
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=your_api_version
```
7. Start Azure MCP Server with SSE
```bash
npx -y @azure/mcp@latest server start --transport sse
```