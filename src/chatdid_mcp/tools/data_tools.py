"""
Data loading and exploration tools for ChatDiD MCP server.
"""

from typing import Any, Dict
from mcp.types import Tool, TextContent
import json
import logging

logger = logging.getLogger(__name__)

# Tool definition for loading data
load_data_tool = Tool(
    name="load_data",
    description="Load dataset for DID analysis from various file formats (CSV, Excel, Stata, Parquet)",
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the data file"
            },
            "file_type": {
                "type": "string", 
                "description": "File type (auto, csv, xlsx, dta, parquet)",
                "default": "auto"
            },
            "sheet_name": {
                "type": "string",
                "description": "Sheet name for Excel files (optional)"
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8)",
                "default": "utf-8"
            }
        },
        "required": ["file_path"]
    }
)

async def load_data_handler(analyzer, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle data loading requests."""
    try:
        file_path = arguments["file_path"]
        file_type = arguments.get("file_type", "auto")
        
        # Prepare kwargs for pandas
        kwargs = {}
        if "sheet_name" in arguments:
            kwargs["sheet_name"] = arguments["sheet_name"]
        if "encoding" in arguments:
            kwargs["encoding"] = arguments["encoding"]
        
        result = await analyzer.load_data(file_path, file_type, **kwargs)
        
        if result["status"] == "success":
            info = result["info"]
            response = f"""
# Data Loading Successful ✅

**Dataset Overview:**
- **Shape:** {info['shape'][0]:,} rows × {info['shape'][1]} columns
- **Columns:** {', '.join(info['columns'][:10])}{'...' if len(info['columns']) > 10 else ''}

**Data Types:**
{chr(10).join([f"- {col}: {dtype}" for col, dtype in list(info['dtypes'].items())[:10]])}

**Missing Values:**
{chr(10).join([f"- {col}: {count}" for col, count in info['missing_values'].items() if count > 0][:5])}

**Next Steps:**
1. Use `explore_data` to analyze panel structure
2. Identify unit, time, outcome, and treatment variables
3. Check for staggered treatment adoption patterns

The data is now loaded and ready for DID analysis!
"""
        else:
            response = f"❌ **Error loading data:** {result['message']}"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error in load_data_handler: {e}")
        return [TextContent(type="text", text=f"❌ **Error:** {str(e)}")]

# Attach handler to tool
load_data_tool.handler = load_data_handler


# Tool definition for exploring data
explore_data_tool = Tool(
    name="explore_data",
    description="Explore loaded dataset to understand panel structure and identify DID variables",
    inputSchema={
        "type": "object",
        "properties": {
            "show_sample": {
                "type": "boolean",
                "description": "Whether to show sample data",
                "default": True
            },
            "analyze_balance": {
                "type": "boolean", 
                "description": "Whether to analyze panel balance",
                "default": True
            }
        },
        "required": []
    }
)

async def explore_data_handler(analyzer, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle data exploration requests."""
    try:
        result = await analyzer.explore_data()
        
        if result["status"] == "error":
            return [TextContent(type="text", text=f"❌ **Error:** {result['message']}")]
        
        exploration = result["exploration"]
        
        # Build exploration report
        response = "# Data Exploration Report 📊\n\n"
        
        # Panel structure analysis
        panel = exploration["panel_structure"]
        response += "## Panel Structure Analysis\n\n"
        response += f"**Potential Unit Variables:** {', '.join(panel['potential_unit_vars']) if panel['potential_unit_vars'] else 'None detected'}\n\n"
        response += f"**Potential Time Variables:** {', '.join(panel['potential_time_vars']) if panel['potential_time_vars'] else 'None detected'}\n\n"
        response += f"**Total Observations:** {panel['n_observations']:,}\n\n"
        
        # Treatment patterns
        treatment = exploration["treatment_patterns"]
        response += "## Treatment Variable Analysis\n\n"
        if treatment["potential_treatment_vars"]:
            response += "**Potential Treatment Variables:**\n"
            for treat in treatment["potential_treatment_vars"]:
                response += f"- `{treat['column']}`: {treat['treatment_share']:.1%} treated\n"
        else:
            response += "No clear binary treatment variables detected.\n"
        
        response += "\n## Recommendations\n\n"
        for rec in exploration["recommendations"]:
            response += f"{rec}\n"
        
        response += "\n**Next Step:** Use `diagnose_twfe` after setting up your variables to check for potential bias."
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error in explore_data_handler: {e}")
        return [TextContent(type="text", text=f"❌ **Error:** {str(e)}")]

# Attach handler to tool
explore_data_tool.handler = explore_data_handler
