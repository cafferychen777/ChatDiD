"""
ChatDiD MCP Server

A Model Context Protocol server for interactive Difference-in-Differences analysis.
Enables users to perform robust DID analysis through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "ChatDiD Team"
__email__ = "team@chatdid.com"

# --- Global initialization (order matters) ---

# 1. Configure matplotlib backend BEFORE any imports to prevent macOS Dock icon
import matplotlib
matplotlib.use('Agg')

# 2. Disable R graphics devices (prevent GUI windows)
import os
os.environ['R_DEFAULT_DEVICE'] = 'pdf'

# 3. Logging: single configuration point for the entire package.
#    MCP STDIO transport requires all log output on stderr.
#    Individual modules should only use logging.getLogger(__name__).
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# 4. Export the FastMCP server instance
from .server import mcp

__all__ = ["mcp"]
