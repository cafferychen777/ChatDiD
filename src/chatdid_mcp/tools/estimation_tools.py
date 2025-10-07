"""
DID estimation tools for ChatDiD MCP server.
"""

from typing import Any, Dict
from mcp.types import Tool, TextContent
import json
import logging

logger = logging.getLogger(__name__)

# Tool definition for DID estimation
estimate_did_tool = Tool(
    name="estimate_did",
    description="Estimate treatment effects using modern heterogeneity-robust DID methods",
    inputSchema={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "DID estimation method",
                "enum": [
                    "callaway_santanna",
                    "sun_abraham", 
                    "borusyak_jaravel_spiess",
                    "gardner_two_stage",
                    "de_chaisemartin_dhaultfoeuille",
                    "twfe_baseline"
                ],
                "default": "callaway_santanna"
            },
            "control_group": {
                "type": "string",
                "description": "Control group for CS estimator",
                "enum": ["nevertreated", "notyettreated"],
                "default": "notyettreated"
            },
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of covariate column names",
                "default": []
            },
            "cluster_var": {
                "type": "string",
                "description": "Variable to cluster standard errors on",
                "default": None
            },
            "anticipation": {
                "type": "integer",
                "description": "Number of periods of anticipation allowed",
                "default": 0
            },
            "event_study": {
                "type": "boolean",
                "description": "Whether to compute event study estimates",
                "default": True
            },
            "max_e": {
                "type": "integer", 
                "description": "Maximum event time to include",
                "default": 10
            },
            "min_e": {
                "type": "integer",
                "description": "Minimum event time to include", 
                "default": -10
            }
        },
        "required": []
    }
)

async def estimate_did_handler(analyzer, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle DID estimation requests."""
    try:
        method = arguments.get("method", "callaway_santanna")
        
        # Run estimation
        result = await analyzer.estimate_did(
            method=method,
            control_group=arguments.get("control_group", "notyettreated"),
            covariates=arguments.get("covariates", []),
            cluster_var=arguments.get("cluster_var"),
            anticipation=arguments.get("anticipation", 0),
            event_study=arguments.get("event_study", True),
            max_e=arguments.get("max_e", 10),
            min_e=arguments.get("min_e", -10)
        )
        
        if result["status"] == "error":
            return [TextContent(type="text", text=f"❌ **Error:** {result['message']}")]
        
        estimates = result["estimates"]
        
        # Build results report
        response = f"# DID Estimation Results ({method.replace('_', ' ').title()}) 📈\n\n"
        
        # Overall ATT
        if "overall_att" in estimates:
            att = estimates["overall_att"]
            response += "## Overall Average Treatment Effect on Treated (ATT)\n\n"
            response += f"**Estimate:** {att['estimate']:.4f}\n"
            response += f"**Standard Error:** {att['se']:.4f}\n"
            response += f"**95% Confidence Interval:** [{att['ci_lower']:.4f}, {att['ci_upper']:.4f}]\n"
            response += f"**P-value:** {att['pvalue']:.4f}\n"
            
            # Interpretation
            if att['pvalue'] < 0.05:
                direction = "positive" if att['estimate'] > 0 else "negative"
                response += f"\n✅ **Statistically significant {direction} treatment effect detected.**\n\n"
            else:
                response += f"\n❌ **No statistically significant treatment effect detected.**\n\n"
        
        # Event study results
        if "event_study" in estimates and arguments.get("event_study", True):
            es = estimates["event_study"]
            response += "## Event Study Results\n\n"
            response += "**Pre-treatment periods (testing parallel trends):**\n"
            
            # Pre-treatment coefficients
            pre_periods = [period for period in es.keys() if period < 0]
            pre_periods.sort()
            
            significant_pre = 0
            for period in pre_periods[-5:]:  # Show last 5 pre-periods
                coef = es[period]
                is_sig = coef['pvalue'] < 0.05
                if is_sig:
                    significant_pre += 1
                sig_marker = "⚠️" if is_sig else "✅"
                response += f"- Period {period}: {coef['estimate']:.4f} ({coef['se']:.4f}) {sig_marker}\n"
            
            response += f"\n**Post-treatment periods:**\n"
            post_periods = [period for period in es.keys() if period >= 0]
            post_periods.sort()
            
            for period in post_periods[:5]:  # Show first 5 post-periods
                coef = es[period]
                is_sig = coef['pvalue'] < 0.05
                sig_marker = "✅" if is_sig else "❌"
                response += f"- Period {period}: {coef['estimate']:.4f} ({coef['se']:.4f}) {sig_marker}\n"
            
            # Parallel trends assessment
            response += f"\n**Parallel Trends Assessment:**\n"
            if significant_pre > len(pre_periods) * 0.3:  # More than 30% significant
                response += "⚠️ **Warning:** Multiple significant pre-treatment coefficients detected.\n"
                response += "- Parallel trends assumption may be violated\n"
                response += "- Consider sensitivity analysis or alternative identification strategies\n"
            else:
                response += "✅ **Pre-treatment coefficients mostly insignificant - parallel trends plausible.**\n"
        
        # Method-specific notes
        response += f"\n## Method Notes: {method.replace('_', ' ').title()}\n\n"
        
        method_notes = {
            "callaway_santanna": "Uses clean 2×2 comparisons, avoiding forbidden comparisons. Robust to heterogeneous treatment effects.",
            "sun_abraham": "Interaction-weighted estimator that corrects dynamic TWFE. Fast and convenient.",
            "borusyak_jaravel_spiess": "Imputation approach using all pre-treatment periods. Efficient under long-run parallel trends.",
            "gardner_two_stage": "Two-stage imputation estimator. Very fast implementation.",
            "de_chaisemartin_dhaultfoeuille": "Handles non-binary and non-absorbing treatments.",
            "twfe_baseline": "Traditional two-way fixed effects. May be biased with heterogeneous effects."
        }
        
        response += method_notes.get(method, "Modern heterogeneity-robust estimator.")
        
        response += "\n\n**Next Steps:**\n"
        response += "1. Use `visualize_results` to create event study plots\n"
        response += "2. Run `sensitivity_analysis` to test robustness\n"
        response += "3. Compare with other estimation methods for robustness\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error in estimate_did_handler: {e}")
        return [TextContent(type="text", text=f"❌ **Error:** {str(e)}")]

# Attach handler to tool
estimate_did_tool.handler = estimate_did_handler
