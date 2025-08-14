from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel

mcp = FastMCP("DataAnalysis")

@mcp.tool()
def test() -> dict:
    return {"ok": True, "message": "pong"}

@mcp.prompt()
def default_prompt(message: str) -> list[base.Message]:
    return [
        base.AssistantMessage(
            "You are a helpful data analysis assistant. \n"
            "Please clearly organize and return the results of the tool calling and the data analysis."
        ),
        base.UserMessage(message),
    ]

if __name__ == "__main__":
    mcp.run(transport="stdio")