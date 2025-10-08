"""
ChatDiD MCP Server

A Model Context Protocol server for interactive Difference-in-Differences analysis.
Enables users to perform robust DID analysis through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "ChatDiD Team"
__email__ = "team@chatdid.com"

# For FastMCP 2.0, the mcp instance is exported from server.py
# FastMCP CLI will automatically discover and use it
from .server import mcp

__all__ = ["mcp"]
