"""
Diagnostic tools for TWFE bias detection in ChatDiD MCP server.
"""

from typing import Any, Dict
from mcp.types import Tool, TextContent
import json
import logging

logger = logging.getLogger(__name__)

# Tool definition for TWFE diagnostics
diagnose_twfe_tool = Tool(
    name="diagnose_twfe",
    description="Diagnose potential bias in Two-Way Fixed Effects (TWFE) estimation using Goodman-Bacon decomposition and negative weighting tests",
    inputSchema={
        "type": "object",
        "properties": {
            "unit_col": {
                "type": "string",
                "description": "Column name for unit identifier"
            },
            "time_col": {
                "type": "string", 
                "description": "Column name for time variable"
            },
            "outcome_col": {
                "type": "string",
                "description": "Column name for outcome variable"
            },
            "treatment_col": {
                "type": "string",
                "description": "Column name for treatment indicator"
            },
            "cohort_col": {
                "type": "string",
                "description": "Column name for treatment cohort (optional for staggered adoption)",
                "default": None
            },
            "run_bacon_decomp": {
                "type": "boolean",
                "description": "Whether to run Goodman-Bacon decomposition",
                "default": True
            },
            "run_twfe_weights": {
                "type": "boolean",
                "description": "Whether to analyze TWFE weights",
                "default": True
            }
        },
        "required": ["unit_col", "time_col", "outcome_col", "treatment_col"]
    }
)

async def diagnose_twfe_handler(analyzer, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle TWFE diagnostic requests."""
    try:
        # Update analyzer configuration
        analyzer.config.update({
            "unit_col": arguments["unit_col"],
            "time_col": arguments["time_col"], 
            "outcome_col": arguments["outcome_col"],
            "treatment_col": arguments["treatment_col"],
            "cohort_col": arguments.get("cohort_col"),
        })
        
        # Run diagnostics
        result = await analyzer.diagnose_twfe(
            run_bacon_decomp=arguments.get("run_bacon_decomp", True),
            run_twfe_weights=arguments.get("run_twfe_weights", True)
        )
        
        if result["status"] == "error":
            return [TextContent(type="text", text=f"❌ **Error:** {result['message']}")]
        
        diagnostics = result["diagnostics"]
        
        # Build diagnostic report
        response = "# TWFE Diagnostic Report 🔍\n\n"
        
        # Basic setup info
        response += "## Analysis Setup\n\n"
        response += f"- **Unit Variable:** `{arguments['unit_col']}`\n"
        response += f"- **Time Variable:** `{arguments['time_col']}`\n"
        response += f"- **Outcome Variable:** `{arguments['outcome_col']}`\n"
        response += f"- **Treatment Variable:** `{arguments['treatment_col']}`\n"
        if arguments.get("cohort_col"):
            response += f"- **Cohort Variable:** `{arguments['cohort_col']}`\n"
        response += "\n"
        
        # Treatment timing analysis
        if "treatment_timing" in diagnostics:
            timing = diagnostics["treatment_timing"]
            response += "## Treatment Timing Analysis\n\n"
            response += f"- **Treatment Type:** {timing['type']}\n"
            response += f"- **Number of Cohorts:** {timing.get('n_cohorts', 'N/A')}\n"
            response += f"- **Staggered Adoption:** {'Yes' if timing['is_staggered'] else 'No'}\n\n"
        
        # Goodman-Bacon decomposition results
        if "bacon_decomp" in diagnostics:
            bacon = diagnostics["bacon_decomp"]
            response += "## Goodman-Bacon Decomposition\n\n"
            response += f"**Overall TWFE Estimate:** {bacon.get('overall_estimate', 'N/A'):.4f}\n\n"
            
            if "comparison_types" in bacon:
                response += "**Comparison Breakdown:**\n"
                for comp_type, details in bacon["comparison_types"].items():
                    response += f"- **{comp_type}:** {details['weight']:.1%} weight, estimate = {details['estimate']:.4f}\n"
                response += "\n"
            
            # Warning about forbidden comparisons
            forbidden_weight = bacon.get("forbidden_comparison_weight", 0)
            if forbidden_weight > 0.1:  # More than 10% weight on forbidden comparisons
                response += "⚠️ **Warning:** High weight on forbidden comparisons detected!\n"
                response += f"- {forbidden_weight:.1%} of weight comes from already-treated units as controls\n"
                response += "- This suggests potential bias in TWFE estimates\n\n"
        
        # TWFE weights analysis
        if "twfe_weights" in diagnostics:
            weights = diagnostics["twfe_weights"]
            response += "## TWFE Weights Analysis\n\n"
            response += f"- **Negative Weights:** {weights.get('negative_weight_share', 0):.1%}\n"
            response += f"- **Robustness Measure:** {weights.get('robustness_measure', 'N/A')}\n\n"
            
            if weights.get("negative_weight_share", 0) > 0.05:  # More than 5% negative weights
                response += "⚠️ **Warning:** Substantial negative weighting detected!\n"
                response += "- TWFE estimates may be severely biased\n"
                response += "- Consider using heterogeneity-robust estimators\n\n"
        
        # Recommendations
        response += "## Recommendations\n\n"
        
        is_problematic = (
            diagnostics.get("bacon_decomp", {}).get("forbidden_comparison_weight", 0) > 0.1 or
            diagnostics.get("twfe_weights", {}).get("negative_weight_share", 0) > 0.05
        )
        
        if is_problematic:
            response += "🚨 **TWFE appears problematic for your data. Recommended actions:**\n\n"
            response += "1. **Use robust estimators:** Try Callaway & Sant'Anna, Sun & Abraham, or imputation estimators\n"
            response += "2. **Run `estimate_did`** with method='callaway_santanna' or method='sun_abraham'\n"
            response += "3. **Conduct sensitivity analysis** to test robustness of findings\n"
            response += "4. **Check parallel trends assumption** with pre-treatment event studies\n"
        else:
            response += "✅ **TWFE appears relatively unproblematic for your data.**\n\n"
            response += "- Low weight on forbidden comparisons\n"
            response += "- Minimal negative weighting\n"
            response += "- Standard TWFE may be appropriate, but consider robust methods for comparison\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error in diagnose_twfe_handler: {e}")
        return [TextContent(type="text", text=f"❌ **Error:** {str(e)}")]

# Attach handler to tool
diagnose_twfe_tool.handler = diagnose_twfe_handler
