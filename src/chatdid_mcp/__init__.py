"""
ChatDiD MCP Server

A Model Context Protocol server for interactive Difference-in-Differences analysis.
Enables users to perform robust DID analysis through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "ChatDiD Team"
__email__ = "team@chatdid.com"

# --- Global initialization (order matters) ---
# Steps 1-3 MUST execute before any submodule import (step 4) because
# submodules trigger matplotlib/R at import time. E402 is intentional.

# 1. Configure matplotlib backend BEFORE any imports to prevent macOS Dock icon
import matplotlib  # noqa: E402
matplotlib.use('Agg')

# 2. Disable R graphics devices (prevent GUI windows)
import os  # noqa: E402
os.environ['R_DEFAULT_DEVICE'] = 'pdf'

# 3. Logging: single configuration point for the entire package.
#    MCP STDIO transport requires all log output on stderr.
#    Individual modules should only use logging.getLogger(__name__).
import logging  # noqa: E402
import sys  # noqa: E402
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# 4. Export the FastMCP server instance
from .server import mcp  # noqa: E402

__all__ = ["mcp"]
