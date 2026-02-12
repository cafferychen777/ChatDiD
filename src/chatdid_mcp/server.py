"""
Main MCP Server for ChatDiD

This module implements the core MCP server that provides tools and resources
for interactive Difference-in-Differences analysis using FastMCP 2.0.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from fastmcp import FastMCP

from .did_analyzer import DiDAnalyzer
from .storage_manager import StorageManager
from .models import SensitivityAnalysisParams

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="chatdid-mcp-server",
    instructions="Interactive Difference-in-Differences analysis through chat",
)

# Global instances (lazy initialization).
# MCP STDIO transport = one process per client session.
# Within a session, load_data() resets all analysis state (diagnostics,
# results, config, workflow) to prevent cross-dataset contamination.
#
# NOTE on async design: MCP tools are declared async to satisfy the FastMCP
# interface, but the heavy work (R estimation via rpy2) is synchronous.
# This is intentional for STDIO transport, which processes one request at a
# time — there is no concurrency to unblock.  Wrapping R calls in
# asyncio.to_thread() would add complexity without benefit and is unsafe
# because rpy2's R interpreter is single-threaded (concurrent calls from
# multiple threads can crash).  If the transport changes to SSE/WebSocket
# in the future, a serializing mutex around rpy2 calls would be needed.
analyzer = None
storage_manager = None


def get_analyzer():
    """Get or create the global analyzer instance."""
    global analyzer
    if analyzer is None:
        analyzer = DiDAnalyzer()
        logger.info("DiD analyzer initialized (NEW INSTANCE)")
    else:
        logger.info(
            f"DiD analyzer reused (existing instance with {len(analyzer.diagnostics)} diagnostics)"
        )
    return analyzer


def get_storage():
    """Get or create the global storage manager instance."""
    global storage_manager
    if storage_manager is None:
        storage_manager = StorageManager()
        logger.info("Storage manager initialized")
    return storage_manager


def _preprocess_did_columns(
    idname: str, tname: str, gname: str, analyzer: Optional[DiDAnalyzer] = None
) -> Dict[str, str]:
    """
    Thin wrapper that delegates to DiDAnalyzer.prepare_data_for_estimation().

    Maps DID estimation parameter names (idname/tname/gname) to the canonical
    preprocessing method, which auto-detects binary treatment vs cohort variables.

    Args:
        idname: Unit identifier column name
        tname: Time variable column name
        gname: Group/cohort/treatment variable column name
        analyzer: DiDAnalyzer instance (if None, uses get_analyzer())

    Returns:
        Dict with actual column names to use:
        - idname: Actual unit column (may be converted)
        - gname: Actual cohort column (may be converted)

    Example:
        >>> cols = _preprocess_did_columns("country", "year", "dem")
        >>> # {"idname": "unit_id_numeric", "gname": "cohort_auto"}
    """
    if analyzer is None:
        analyzer = get_analyzer()

    # Let errors propagate — each MCP tool has its own try/except that
    # formats a user-facing message.  Silently falling back to raw column
    # names would mask real problems (missing columns, wrong data types)
    # and produce misleading estimation results.
    processed = analyzer.prepare_data_for_estimation(
        unit_col=idname, time_col=tname, treatment_col=gname
    )
    return {"idname": processed["unit_col"], "gname": processed["cohort_col"]}


# ---------------------------------------------------------------------------
# Guard helpers — single source of truth for common pre-condition checks.
# Each returns an error string (to be returned from the tool) or None.
# Usage:  if err := _require_data(): return err
# ---------------------------------------------------------------------------


def _require_data() -> Optional[str]:
    """Guard: return error string if no data loaded, else None."""
    if get_analyzer().data is None:
        return "Error: No data loaded. Please load data first."
    return None


def _require_r() -> Optional[str]:
    """Guard: return error string if R integration not available, else None."""
    if not get_analyzer().r_estimators:
        return "Error: R integration not available. Please install rpy2 and required R packages."
    return None


def _require_results() -> Optional[str]:
    """Guard: return error string if no estimation results available, else None."""
    analyzer = get_analyzer()
    if not analyzer.results or "latest" not in analyzer.results:
        return "Error: No estimation results available. Please run an estimator first."
    return None


def _check_result(result: Dict[str, Any]) -> Optional[str]:
    """Guard: return error string if result status is not success, else None."""
    if result.get("status") != "success":
        return f"Error: {result.get('message', 'Unknown error')}"
    return None


def _store_estimation(method_name: str, result: Dict[str, Any]) -> None:
    """Store estimation result and set it as the primary (latest) result.

    The 'latest' key points to the primary estimator's result.
    Workflow steps 4-5 (inference, sensitivity) use this as input.
    """
    analyzer = get_analyzer()
    analyzer.results[method_name] = result
    analyzer.results["latest"] = result
    logger.info(f"Stored {method_name} estimation results")


# ---------------------------------------------------------------------------
# Response formatting helpers — single source of truth for common output
# blocks across estimation tools.  Changes to format, thresholds, or
# terminology only need to happen here.
# ---------------------------------------------------------------------------


def _format_overall_att(
    result: Dict[str, Any],
    label: str = "Overall Average Treatment Effect",
) -> str:
    """Format the Overall ATT section.  Returns empty string if absent."""
    if "overall_att" not in result:
        return ""
    overall = result["overall_att"]
    parts = [f"{label}\n"]
    parts.append(f"- Estimate: {overall['estimate']:.4f}")
    if overall.get("se") is not None:
        parts.append(f"- Std. Error: {overall['se']:.4f}")
    if overall.get("ci_lower") is not None and overall.get("ci_upper") is not None:
        parts.append(f"- 95% CI: [{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}]")
    if overall.get("pvalue") is not None:
        parts.append(f"- p-value: {overall['pvalue']:.4f}")
    return "\n".join(parts) + "\n\n"


def _format_event_study_table(
    result: Dict[str, Any],
    show_ci: bool = True,
) -> str:
    """Format event study estimates as a markdown table."""
    event_study = result.get("event_study")
    if not event_study:
        return ""
    lines = ["Event Study Estimates\n"]
    if show_ci:
        lines.append("| Event Time | Estimate | Std. Error | 95% CI | p-value |")
        lines.append("|------------|----------|------------|--------|----------|")
        for e, est in sorted(event_study.items()):
            lines.append(
                f"| {e:^10} | {est['estimate']:^8.4f} | {est['se']:^10.4f} | "
                f"[{est['ci_lower']:.3f}, {est['ci_upper']:.3f}] | {est['pvalue']:.4f} |"
            )
    else:
        lines.append("| Event Time | Estimate | Std. Error | p-value |")
        lines.append("|------------|----------|------------|----------|")
        for e, est in sorted(event_study.items()):
            lines.append(
                f"| {e} | {est['estimate']:.4f} | {est['se']:.4f} | {est['pvalue']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _format_pretrends_check(result: Dict[str, Any]) -> str:
    """Check pre-period event study estimates for significant effects."""
    event_study = result.get("event_study")
    if not event_study:
        return ""
    pre_period = [est for e, est in event_study.items() if e < 0]
    if not pre_period:
        return ""
    sig_count = sum(1 for est in pre_period if est["pvalue"] < 0.05)
    if sig_count > 0:
        return (
            f"\nWarning: {sig_count} pre-treatment periods show significant effects\n"
            "This may indicate violations of parallel trends assumption.\n"
        )
    return "\nNo significant pre-trends detected\n"


def _format_significance(result: Dict[str, Any]) -> str:
    """Format statistical significance interpretation of overall ATT."""
    overall = result.get("overall_att")
    if not overall:
        return ""
    pvalue = overall.get("pvalue")
    if pvalue is not None:
        significant = pvalue < 0.05
    elif overall.get("se") is not None and overall["se"] != 0:
        # Fallback: compute z-stat when p-value is not available (gsynth, synthdid)
        z_stat = abs(overall["estimate"] / overall["se"])
        significant = z_stat > 1.96
    else:
        return ""
    if significant:
        return "\nStatistically significant treatment effect detected\n"
    return "\nNo statistically significant treatment effect at 5% level\n"


# =============================================================================
# TOOLS - Using FastMCP decorators
# =============================================================================


@mcp.tool()
async def load_data(
    file_path: str,
    file_type: str = "auto",
    sheet_name: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """
    Load dataset for DID analysis from various file formats.

    Args:
        file_path: Path to the data file
        file_type: File type (auto, csv, xlsx, dta, parquet)
        sheet_name: Sheet name for Excel files (optional)
        encoding: File encoding (default: utf-8)

    Returns:
        Summary of the loaded data including dataset info
    """
    try:
        # Prepare kwargs for loading
        kwargs = {"encoding": encoding}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name

        # Directly await the async method
        result = await get_analyzer().load_data(file_path, file_type, **kwargs)

        if result["status"] == "success":
            info = result["info"]
            return f"""
# Data Loading Successful

Dataset Overview:
- Shape: {info["shape"][0]:,} rows × {info["shape"][1]} columns
- Columns: {", ".join(info["columns"][:10])}{"..." if len(info["columns"]) > 10 else ""}

Next Steps:
1. Use explore_data to analyze panel structure
2. Identify unit, time, outcome, and treatment variables
3. Check for staggered treatment adoption patterns

The data is now loaded and ready for DID analysis!
"""
        else:
            return f"Error loading data: {result['message']}"

    except FileNotFoundError as e:
        logger.error(f"File not found in load_data: {e}")
        return f"""
Error: File not found

The specified file does not exist: {file_path}

Common Causes:

1. Using relative paths [Not supported]
   - MCP servers run in a different directory
   - Relative paths like data/file.csv will not work

2. Using tilde (~) in paths [Not supported]
   - Tilde expansion may not work in MCP context
   - ~/ChatDiD/data.csv will fail

3. Attaching files to chat [Not supported]
   - File attachments are not supported by MCP servers
   - Files must exist on your filesystem

Solution: Use absolute paths

Get the absolute path of your file:


# Method 1: Use realpath
realpath data/examples/mpdta.csv

# Method 2: Use pwd + filename
echo "$(pwd)/data/examples/mpdta.csv"


Then use the full path in your request:

Load /Users/yourname/projects/ChatDiD/data/examples/mpdta.csv


See README.md "Usage Example" section for more details about file paths.
"""

    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        # Check if error message indicates file/path issues
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "no such file",
                "file not found",
                "does not exist",
                "cannot find",
            ]
        ):
            return f"""
Error: Cannot access file

{str(e)}

This looks like a file path issue.

Solution: Use absolute paths

MCP servers require absolute file paths. Get the absolute path:


# Method 1: Use realpath
realpath your_file.csv

# Method 2: Use pwd
echo "$(pwd)/your_file.csv"


Then use the full path like: /Users/yourname/projects/ChatDiD/data/file.csv

See README.md "Usage Example" section for details about file paths.
"""
        else:
            return f"Error: {str(e)}"


@mcp.tool()
async def explore_data(show_sample: bool = True, analyze_balance: bool = True) -> str:
    """
    Explore loaded dataset to understand panel structure and identify DID variables.

    Args:
        show_sample: Whether to show sample data
        analyze_balance: Whether to analyze panel balance

    Returns:
        Detailed exploration report of the dataset
    """
    try:
        result = await get_analyzer().explore_data()

        if result["status"] == "error":
            return f"Error: {result['message']}"

        exploration = result["exploration"]

        # Build exploration report
        response = "# Data Exploration Report\n\n"

        # Panel structure analysis
        panel = exploration["panel_structure"]
        response += "Panel Structure Analysis\n\n"
        response += f"Potential Unit Variables: {', '.join(panel['potential_unit_vars']) if panel['potential_unit_vars'] else 'None detected'}\n\n"
        response += f"Potential Time Variables: {', '.join(panel['potential_time_vars']) if panel['potential_time_vars'] else 'None detected'}\n\n"
        response += f"Total Observations: {panel['n_observations']:,}\n\n"

        # Treatment patterns
        treatment = exploration["treatment_patterns"]
        response += "Treatment Variable Analysis\n\n"
        if treatment["potential_treatment_vars"]:
            response += "Potential Treatment Variables:\n"
            for treat in treatment["potential_treatment_vars"]:
                response += (
                    f"- {treat['column']}: {treat['treatment_share']:.1%} treated\n"
                )
        else:
            response += "No clear binary treatment variables detected.\n"

        response += "\nRecommendations\n\n"
        for rec in exploration["recommendations"]:
            response += f"{rec}\n"

        response += "\nNext Step: Use diagnose_twfe after setting up your variables to check for potential bias."

        return response

    except Exception as e:
        logger.error(f"Error in explore_data: {e}")
        return f"Error: {str(e)}"


# =============================================================================
# RESOURCES - Analysis guides and results
# =============================================================================


@mcp.resource("chatdid://guide/analysis-workflow")
async def get_analysis_workflow() -> str:
    """DID Analysis Workflow Guide"""
    return """
# DID Analysis Workflow Guide
 Step-by-Step Process
 1. Data Preparation
- Load your panel dataset
- Identify unit, time, outcome, and treatment variables
- Check for missing values and data quality issues
 2. Exploratory Analysis
- Examine treatment timing patterns
- Check for staggered adoption
- Visualize pre-treatment trends
 3. TWFE Diagnostics
- Run Goodman-Bacon decomposition
- Check for negative weighting
- Assess forbidden comparisons
 4. Robust Estimation
- Choose appropriate estimator based on diagnostics
- Estimate treatment effects
- Generate event study plots
 5. Sensitivity Analysis
- Test parallel trends assumption
- Conduct robustness checks
- Validate findings
 6. Interpretation
- Assess statistical and economic significance
- Consider policy implications
- Document limitations
"""


# Note: FastMCP may not yet support resource_list() and uri_template
# These are commented out until FastMCP adds support
# For now, use the manage_storage tool to list and access files

# @mcp.resource_list()
# async def list_outputs() -> List[Dict[str, Any]]:
#     """List all available output files as MCP resources."""
#     storage = get_storage()
#     return storage.list_resources()

# @mcp.resource(uri_template="file:///{path}")
# async def get_output_file(path: str) -> Dict[str, Any]:
#     """Get a specific output file as an MCP resource."""
#     storage = get_storage()
#     uri = f"file:///{path}"
#     return storage.get_resource(uri)


# =============================================================================
# PROMPTS - Interactive templates
# =============================================================================


@mcp.prompt()
def did_analysis_start(research_question: str) -> str:
    """Start a new DID analysis session"""
    return f"""
# Starting DID Analysis for: {research_question}

I'll help you conduct a robust Difference-in-Differences analysis. Let's start by:

1. Loading your data: Use the load_data tool to import your dataset
2. Exploring the data: Use explore_data to understand your panel structure
3. Diagnosing TWFE issues: Use diagnose_twfe to check for bias
4. Choosing robust estimators: Based on diagnostics, select appropriate methods
5. Sensitivity analysis: Validate your results with robustness checks

What type of data do you have, and what treatment are you studying?
"""


# =============================================================================
# 5-STEP WORKFLOW TOOLS
# =============================================================================


@mcp.tool()
async def workflow(
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    cohort_col: Optional[str] = None,
    method: str = "auto",
    cluster_level: Optional[str] = None,
) -> str:
    """
    Execute the complete 5-step DID workflow automatically.

    This follows Roth et al. (2023) best practices:
    1. Check for staggered treatment
    2. Diagnose TWFE bias (if staggered)
    3. Apply robust estimator + robustness check
    4. Assess parallel trends
    5. Finalize inference

    Args:
        unit_col: Column name for unit identifier
        time_col: Column name for time variable
        outcome_col: Column name for outcome variable
        treatment_col: Column name for treatment indicator
        cohort_col: Column name for treatment cohort (optional)
        method: Estimation method ("auto", "callaway_santanna", "sun_abraham",
                "imputation_bjs", "gardner", "dcdh", "gsynth", "synthdid",
                "drdid", "etwfe")
                Note: "efficient" is DISABLED due to systematic issues.
        cluster_level: Variable for clustering standard errors

    Returns:
        Complete DID analysis report

    Note:
        Results Storage: Workflow results are stored with "workflow_" prefix.
        - Primary method results: "workflow_{method}" (e.g., "workflow_imputation_bjs")
        - Robustness check results: "workflow_{robustness_method}"
        - Primary for inference: "latest" (points to the primary estimator;
          Steps 4-5 read this for parallel trends assessment and final inference)

        Export Results: Use export_results(results_key="latest") or
        export_results(results_key="workflow_imputation_bjs") to access workflow results.
    """
    try:
        # Update analyzer config
        get_analyzer().config.update(
            {
                "unit_col": unit_col,
                "time_col": time_col,
                "outcome_col": outcome_col,
                "treatment_col": treatment_col,
                "cohort_col": cohort_col,
            }
        )

        # Run complete workflow
        results = await get_analyzer().workflow.run_complete_workflow(
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            cohort_col=cohort_col,
            method=method,
            cluster_level=cluster_level,
        )

        if results["status"] == "complete":
            return results["final_report"]
        else:
            return f"Workflow incomplete. Last successful step: {results.get('last_step', 'None')}"

    except Exception as e:
        logger.error(f"Error in DID workflow: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def diagnose_twfe(
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    cohort_col: Optional[str] = None,
    run_bacon_decomp: bool = True,
    run_twfe_weights: bool = True,
) -> str:
    """
    Diagnose potential TWFE bias in staggered DID designs.

    Performs comprehensive diagnostics including:
    - Goodman-Bacon decomposition to identify problematic comparisons
    - Negative weights analysis to detect bias
    - Treatment timing variation assessment
    - Recommendations for appropriate estimators

    Args:
        unit_col: Column name for unit identifier
        time_col: Column name for time variable
        outcome_col: Column name for outcome variable
        treatment_col: Column name for treatment indicator
        cohort_col: Column name for treatment cohort (optional)
        run_bacon_decomp: Whether to run Goodman-Bacon decomposition (default: True)
        run_twfe_weights: Whether to analyze TWFE weights (default: True)

    Returns:
        Comprehensive TWFE diagnostic report with recommendations

    Examples:
        >>> # Basic TWFE diagnostics
        >>> diagnose_twfe("county", "year", "employment", "treatment")

        >>> # With cohort specification
        >>> diagnose_twfe("county", "year", "employment", "treatment", "first_treat")

        >>> # Only run Bacon decomposition
        >>> diagnose_twfe("county", "year", "employment", "treatment",
        ...               run_twfe_weights=False)
    """
    try:
        if err := _require_data(): return err
        analyzer = get_analyzer()

        # Update analyzer configuration
        analyzer.config.update(
            {
                "unit_col": unit_col,
                "time_col": time_col,
                "outcome_col": outcome_col,
                "treatment_col": treatment_col,
                "cohort_col": cohort_col,
            }
        )

        # Run diagnostics
        result = await analyzer.diagnose_twfe(
            run_bacon_decomp=run_bacon_decomp, run_twfe_weights=run_twfe_weights
        )

        if result["status"] == "error":
            return f"Error: {result['message']}"

        diagnostics = result["diagnostics"]

        # Store diagnostic results for visualization
        if "bacon_decomp" in diagnostics:
            analyzer.diagnostics["bacon_decomp"] = diagnostics["bacon_decomp"]
        if "twfe_weights" in diagnostics:
            analyzer.diagnostics["twfe_weights"] = diagnostics["twfe_weights"]
        logger.info(
            f"Stored TWFE diagnostic results. Current diagnostics keys: {list(analyzer.diagnostics.keys())}"
        )
        logger.info(f"Analyzer instance ID: {id(analyzer)}")

        # Build diagnostic report
        response = "# TWFE Diagnostic Report\n\n"

        # Basic setup info
        response += "Analysis Setup\n\n"
        response += f"- Unit Variable: {unit_col}\n"
        response += f"- Time Variable: {time_col}\n"
        response += f"- Outcome Variable: {outcome_col}\n"
        response += f"- Treatment Variable: {treatment_col}\n"
        if cohort_col:
            response += f"- Cohort Variable: {cohort_col}\n"
        response += "\n"

        # Treatment timing analysis
        if "treatment_timing" in diagnostics:
            timing = diagnostics["treatment_timing"]
            response += "Treatment Timing Analysis\n\n"
            response += f"- Treatment Type: {timing['type']}\n"
            response += f"- Number of Cohorts: {timing.get('n_cohorts', 'N/A')}\n"
            response += f"- Staggered Adoption: {'Yes' if timing['is_staggered'] else 'No'}\n\n"

        # Goodman-Bacon decomposition results
        if "bacon_decomp" in diagnostics:
            bacon = diagnostics["bacon_decomp"]
            response += "Goodman-Bacon Decomposition\n\n"
            response += f"Overall TWFE Estimate: {bacon.get('overall_estimate', 'N/A'):.4f}\n\n"

            if "comparison_types" in bacon:
                response += "Comparison Breakdown:\n"
                for comp_type, details in bacon["comparison_types"].items():
                    response += f"- {comp_type}: {details['weight']:.1%} weight, estimate = {details['estimate']:.4f}\n"
                response += "\n"

            # Warning about forbidden comparisons
            forbidden_weight = bacon.get("forbidden_comparison_weight", 0)
            if forbidden_weight > 0.1:  # More than 10% weight on forbidden comparisons
                response += (
                    "Warning: High weight on forbidden comparisons detected!\n"
                )
                response += f"- {forbidden_weight:.1%} of weight comes from already-treated units as controls\n"
                response += "- This suggests potential bias in TWFE estimates\n\n"

        # TWFE weights analysis
        if "twfe_weights" in diagnostics:
            weights = diagnostics["twfe_weights"]
            response += "TWFE Weights Analysis\n\n"
            response += f"- Negative Weights: {weights.get('negative_weight_share', 0):.1%}\n"
            response += f"- Robustness Measure: {weights.get('robustness_measure', 'N/A')}\n\n"

            if (
                weights.get("negative_weight_share", 0) > 0.05
            ):  # More than 5% negative weights
                response += "Warning: Substantial negative weighting detected!\n"
                response += "- TWFE estimates may be severely biased\n"
                response += "- Consider using heterogeneity-robust estimators\n\n"

        # Recommendations
        response += "Recommendations\n\n"

        is_problematic = (
            diagnostics.get("bacon_decomp", {}).get("forbidden_comparison_weight", 0)
            > 0.1
            or diagnostics.get("twfe_weights", {}).get("negative_weight_share", 0)
            > 0.05
        )

        if is_problematic:
            response += (
                "TWFE appears problematic for your data. Recommended actions:\n\n"
            )
            response += "1. Use robust estimators: Try Callaway & Sant'Anna, Sun & Abraham, or imputation estimators\n"
            response += "2. Run specific estimator: e.g., estimate_callaway_santanna or estimate_sun_abraham\n"
            response += "3. Conduct sensitivity analysis: Use sensitivity_analysis to test robustness\n"
            response += "4. Check parallel trends: Run power_analysis for pre-trends testing power\n"
        else:
            response += (
                "TWFE appears relatively unproblematic for your data.\n\n"
            )
            response += "- Low weight on forbidden comparisons\n"
            response += "- Minimal negative weighting\n"
            response += "- Standard TWFE may be appropriate, but consider robust methods for comparison\n"

        return response

    except Exception as e:
        logger.error(f"Error in diagnose_twfe: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def diagnose_goodman_bacon(
    formula: str,
    id_var: str,
    time_var: str,
    cohort_col: Optional[str] = None,
) -> str:
    """
    Run Goodman-Bacon (2021) decomposition using bacondecomp::bacon().

    What it checks: FORBIDDEN COMPARISONS in staggered DID designs
    - Detects when treated units at different times are compared to each other
    - Shows weight distribution across comparison types
    - Identifies "Earlier vs Later" and "Later vs Earlier" comparisons

    Important: This is ONE of TWO diagnostic checks you should run:
    1. This tool → Checks for forbidden comparisons (timing issues)
    2. analyze_twfe_weights() → Checks for negative weights

    Both can cause TWFE bias, but they are DIFFERENT problems!

    Args:
        formula: R formula (e.g., "outcome ~ treatment")
        id_var: Unit identifier column
        time_var: Time identifier column
        cohort_col: Cohort variable (first treatment time). If omitted, the
            treatment variable from the formula is auto-detected as binary
            treatment or cohort via prepare_data_for_estimation().

    Returns:
        Decomposition results showing weights and estimates for each 2×2 comparison

    Next Steps:
        - Run analyze_twfe_weights() to complete diagnostic analysis
        - Use create_diagnostic_plots() to visualize results
        - If bias detected, use robust estimators (CS, SA, BJS, Gardner)

    Example:
        >>> diagnose_goodman_bacon(
        ...     formula="lemp ~ treat",
        ...     id_var="countyreal",
        ...     time_var="year"
        ... )
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Extract treatment variable from formula to feed into preprocessing.
        import re
        match = re.search(r'~\s*([\w.]+)', formula)
        if not match:
            return "Error: Could not parse treatment variable from formula. Expected 'outcome ~ treatment'."
        treatment_var = match.group(1)

        # Preprocess through the single source of truth — resolves the
        # canonical cohort column, normalizes never-treated encoding to 0,
        # and ensures numeric unit IDs.
        processed = _preprocess_did_columns(
            idname=id_var, tname=time_var, gname=cohort_col or treatment_var
        )
        actual_cohort_col = processed["gname"]

        # Run Goodman-Bacon decomposition with preprocessed columns
        result = get_analyzer().r_estimators.goodman_bacon_decomposition(
            data=get_analyzer().data,
            formula=formula,
            id_var=processed["idname"],
            time_var=time_var,
            cohort_col=actual_cohort_col,
        )

        if err := _check_result(result): return err

        # Store diagnostic results for visualization
        analyzer_instance = get_analyzer()
        analyzer_instance.diagnostics["bacon_decomp"] = result
        logger.info(
            f"Stored Goodman-Bacon diagnostic results. Current diagnostics keys: {list(analyzer_instance.diagnostics.keys())}"
        )
        logger.info(f"Analyzer instance ID: {id(analyzer_instance)}")

        # Format results
        response = "# Goodman-Bacon Decomposition Results\n\n"
        response += f"Overall TWFE Estimate: {result['overall_estimate']:.4f}\n\n"
        response += "Comparison Breakdown\n\n"

        for comp_type, details in result["comparison_types"].items():
            response += f"{comp_type}:\n"
            response += f"- Weight: {details['weight']:.1%}\n"
            response += f"- Estimate: {details['estimate']:.4f}\n\n"

        # Warning if forbidden comparisons detected
        response += "Diagnosis\n\n"

        if result["forbidden_comparison_weight"] > 0.1:
            response += "WARNING: Forbidden comparisons detected!\n\n"
            response += f'- {result["forbidden_comparison_weight"]:.1%} of weight from "Earlier vs Later" / "Later vs Earlier" comparisons\n'
            response += (
                "- These comparisons mix treatment effects from different periods\n"
            )
            response += (
                "- TWFE estimates likely biased due to treatment timing issues\n\n"
            )
            response += "What this means:\n"
            response += "- Units treated at different times are being compared\n"
            response += (
                "- This violates the assumption needed for valid TWFE estimation\n\n"
            )
            response += "Recommendation: Use heterogeneity-robust estimators designed for staggered adoption\n\n"
            response += "Note: This checks forbidden comparisons. Also run analyze_twfe_weights() to check for negative weights.\n"
        else:
            response += "No forbidden comparisons detected.\n\n"
            response += (
                "- Comparisons are primarily between treated and never-treated units\n"
            )
            response += "- This specific bias source is minimal\n\n"
            response += "Note: This only checks forbidden comparisons. Also run analyze_twfe_weights() to check for negative weights.\n"

        return response

    except Exception as e:
        logger.error(f"Error in Goodman-Bacon decomposition: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def analyze_twfe_weights(
    outcome_col: str, unit_col: str, time_col: str, treatment_col: str
) -> str:
    """
    Analyze negative weights in TWFE estimation.

    What it checks: NEGATIVE WEIGHTS in TWFE regression
    - Detects observations that receive negative weights
    - Negative weights can flip the sign of treatment effects
    - Shows how many observations are affected

    Important: This is ONE of TWO diagnostic checks you should run:
    1. diagnose_goodman_bacon() → Checks for forbidden comparisons (timing issues)
    2. This tool → Checks for negative weights

    Both can cause TWFE bias, but they are DIFFERENT problems!

    Key distinction:
    - Forbidden comparisons: Wrong units being compared (can have positive weights)
    - Negative weights: Weights with wrong sign (< 0)

    Args:
        outcome_col: Column name for outcome variable
        unit_col: Column name for unit identifier
        time_col: Column name for time variable
        treatment_col: Column name for treatment indicator

    Returns:
        Detailed weight analysis showing negative weight shares and affected groups

    Next Steps:
        - Run diagnose_goodman_bacon() to complete diagnostic analysis
        - Use create_diagnostic_plots() to visualize results
        - If bias detected, use robust estimators (CS, SA, BJS, Gardner)

    Examples:
        >>> # Basic weight analysis
        >>> analyze_twfe_weights("lemp", "countyreal", "year", "treat")

        >>> # For panel data
        >>> analyze_twfe_weights("outcome", "unit", "time", "treatment")

    Note:
        Even if negative weights = 0%, forbidden comparisons may still exist!
        Always run BOTH diagnostic tools for complete TWFE bias assessment.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err
        analyzer = get_analyzer()

        # Call R estimator for weights analysis
        result = analyzer.r_estimators.twfe_weights_analysis(
            data=analyzer.data,
            outcome_col=outcome_col,
            unit_col=unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
        )

        if err := _check_result(result): return err

        # Format results
        response = "# TWFE Weights Analysis Report\n\n"

        response += "Analysis Setup\n\n"
        response += f"- Outcome: {outcome_col}\n"
        response += f"- Unit: {unit_col}\n"
        response += f"- Time: {time_col}\n"
        response += f"- Treatment: {treatment_col}\n\n"

        # Overall statistics
        response += "Weight Distribution\n\n"
        response += f"- Negative Weight Share: {result.get('negative_weight_share', 0):.2%}\n"
        response += (
            f"- Number of Negative Weights: {result.get('n_negative_weights', 0)}\n"
        )
        if result.get("negative_weight_share", 0) > 0:
            response += f"- Positive Weight Share: {(1 - result.get('negative_weight_share', 0)):.2%}\n"
        else:
            response += "- Positive Weight Share: 100.00%\n"
        response += "\n"

        # Robustness measure
        if "robustness_measure" in result and result["robustness_measure"] is not None:
            response += "Robustness Assessment\n\n"
            response += (
                f"Robustness Parameter (σ): {result['robustness_measure']:.4f}\n\n"
            )

            if result["robustness_measure"] < 0.5:
                response += (
                    "High Robustness: TWFE estimates are relatively stable\n"
                )
            elif result["robustness_measure"] < 1.0:
                response += "Moderate Robustness: Some sensitivity to treatment effect heterogeneity.\n"
            else:
                response += "Low Robustness: TWFE estimates highly sensitive to heterogeneity.\n"
            response += "\n"

        # Detailed weight breakdown
        if "weight_details" in result:
            response += "Weight Details by Group\n\n"
            details = result["weight_details"]

            if "by_cohort" in details:
                response += "By Treatment Cohort\n\n"
                for cohort, info in details["by_cohort"].items():
                    response += f"Cohort {cohort}:\n"
                    response += f"- Observations: {info['n']}\n"
                    response += f"- Average Weight: {info['avg_weight']:.4f}\n"
                    response += f"- Negative Weight Share: {info['neg_share']:.2%}\n\n"

            if "by_period" in details:
                response += "By Time Period\n\n"
                for period, info in details["by_period"].items():
                    response += f"Period {period}:\n"
                    response += f"- Observations: {info['n']}\n"
                    response += f"- Average Weight: {info['avg_weight']:.4f}\n"
                    response += f"- Negative Weight Share: {info['neg_share']:.2%}\n\n"

        # Interpretation and recommendations
        response += "Interpretation\n\n"

        neg_share = result.get("negative_weight_share", 0)
        if neg_share < 0.01:
            response += "No negative weights detected.\n\n"
            response += (
                "- No observations receive negative weights in TWFE regression\n"
            )
            response += "- This specific source of bias is not present\n\n"
            response += "Note: This only checks for negative weights. Other bias sources may exist:\n"
            response += "- Run diagnose_goodman_bacon() to check for forbidden comparisons\n"
            response += (
                "- Forbidden comparisons can cause bias even without negative weights\n"
            )
        elif neg_share < 0.05:
            response += "Some negative weighting detected\n\n"
            response += f"- {neg_share:.1%} of weights are negative\n"
            response += (
                "- These negative weights can cause bias when treatment effects vary\n"
            )
            response += "- Consider robust estimators for comparison\n\n"
            response += "Also check: Run diagnose_goodman_bacon() for forbidden comparisons analysis\n"
        else:
            response += "Substantial negative weighting detected\n\n"
            response += f"- {neg_share:.1%} of weights are negative\n"
            response += "- TWFE estimates likely biased due to negative weighting\n"
            response += "- Strongly recommend using heterogeneity-robust estimators:\n"
            response += "  - Callaway & Sant'Anna (estimate_callaway_santanna)\n"
            response += "  - Sun & Abraham (estimate_sun_abraham)\n"
            response += "  - Imputation methods (estimate_bjs_imputation)\n\n"
            response += "Also check: Run diagnose_goodman_bacon() for additional bias sources\n"

        # Store diagnostic results for visualization
        analyzer.diagnostics["twfe_weights"] = result
        logger.info(
            f"Stored TWFE weights diagnostic results. Current diagnostics keys: {list(analyzer.diagnostics.keys())}"
        )
        logger.info(f"Analyzer instance ID: {id(analyzer)}")

        return response

    except Exception as e:
        logger.error(f"Error in analyze_twfe_weights: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_callaway_santanna(
    yname: str,
    tname: str,
    idname: str,
    gname: str,
    control_group: str = "notyettreated",
    xformla: Optional[str] = None,
    anticipation: int = 0,
) -> str:
    """
    Callaway & Sant'Anna (2021) doubly robust DID estimator.

    This estimator is robust to heterogeneous treatment effects and uses
    either not-yet-treated or never-treated units as controls.

    Args:
        yname: Outcome variable name
        tname: Time variable name
        idname: Unit identifier name
        gname: Group/cohort variable (first treatment time)
        control_group: "notyettreated" or "nevertreated"
        xformla: Covariate formula (e.g., "~ X1 + X2")
        anticipation: Number of periods before treatment with anticipation effects

    Returns:
        Event study estimates with overall ATT
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data: delegates to the single source of truth
        processed = _preprocess_did_columns(idname, tname, gname)
        actual_idname = processed["idname"]
        actual_gname = processed["gname"]

        # Run Callaway & Sant'Anna estimation
        result = get_analyzer().r_estimators.callaway_santanna_estimator(
            data=get_analyzer().data,
            yname=yname,
            tname=tname,
            idname=actual_idname,
            gname=actual_gname,
            control_group=control_group,
            xformla=xformla,
        )

        if err := _check_result(result): return err

        _store_estimation("callaway_santanna", result)

        # Format results
        response = "# Callaway & Sant'Anna (2021) Results\n\n"
        response += f"Method: Doubly Robust DID\n"
        response += f"Control Group: {result['control_group']}\n"
        response += f"Aggregation: Group (official recommendation)\n\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result)
        response += _format_pretrends_check(result)
        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in Callaway & Sant'Anna estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_sun_abraham(formula: str, cluster_var: Optional[str] = None) -> str:
    """
    Sun & Abraham (2021) interaction-weighted estimator using direct formula.

    Uses the fixest package with sunab() to estimate heterogeneity-robust
    treatment effects for staggered adoption designs.

    Args:
        formula: R formula with sunab() function - passed directly to fixest::feols()
                Examples:
                - "lemp ~ sunab(first.treat, year) | countyreal + year" (with dots)
                - "outcome ~ x1 + sunab(cohort, time) | unit + time" (with covariates)
                - "y ~ sunab(g, t) | i + t" (simple names)
        cluster_var: Variable for clustering standard errors (optional)

    Returns:
        Event study estimates with clustered standard errors

    Note:
        This method directly passes your formula to R's fixest package without
        parameter parsing, supporting any valid fixest formula syntax including
        column names with dots, dashes, or other special characters.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Validate formula contains sunab() function
        if "sunab(" not in formula:
            return "Formula must contain sunab() function. Example: 'lemp ~ sunab(first.treat, year) | countyreal + year'"

        # Run Sun & Abraham estimation with direct formula (no parsing)
        result = get_analyzer().r_estimators.sun_abraham_estimator(
            data=get_analyzer().data, formula=formula, cluster_var=cluster_var
        )

        if err := _check_result(result): return err

        _store_estimation("sun_abraham", result)

        # Format results
        response = "# Sun & Abraham (2021) Results\n\n"
        response += "Method: Interaction-Weighted Estimator\n\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result)
        response += _format_pretrends_check(result)
        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in Sun & Abraham estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_bjs_imputation(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,  # REQUIRED - not Optional
    horizon: int = 10,
    pretrends_test: bool = True,
) -> str:
    """
    Borusyak, Jaravel & Spiess (2024) imputation estimator.

    Uses the didimputation approach for efficient computation with
    staggered treatment timing. Robust to heterogeneous treatment effects.

    IMPORTANT: BJS requires a COHORT variable (first treatment time), NOT a
    binary 0/1 treatment indicator!

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        cohort_col: Cohort/group variable indicating FIRST TREATMENT TIME
                   (e.g., 'first.treat' with values 0, 2004, 2006, 2007)
                   Value 0 indicates never-treated units
                   REQUIRED for proper BJS estimation
        horizon: Number of periods to estimate after treatment (default: 10)
        pretrends_test: Whether to perform pre-trends test (default: True)

    Returns:
        Event study estimates with ATT and pre-trends analysis

    Examples:
        >>> # Correct usage with cohort variable
        >>> estimate_bjs_imputation("lemp", "countyreal", "year", "first.treat")

        >>> # Another example
        >>> estimate_bjs_imputation("outcome", "unit", "time", "cohort")

    Reference:
        Borusyak, Jaravel & Spiess (2024) "Revisiting Event Study Designs:
        Robust and Efficient Estimation"
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, cohort_col)
        actual_unit_col = processed["idname"]
        actual_cohort_col = processed["gname"]

        # Run BJS Imputation estimation
        result = get_analyzer().r_estimators.bjs_imputation_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            cohort_col=actual_cohort_col,
            horizon=horizon,
            pretrends_test=pretrends_test,
        )

        if err := _check_result(result): return err

        _store_estimation("bjs_imputation", result)

        # Format results
        response = "# Borusyak, Jaravel & Spiess (2024) Results\n\n"
        response += f"Method: Imputation Estimator\n"
        response += f"Horizon: {horizon} periods\n\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result)

        # BJS-specific formal pre-trends test
        if pretrends_test and "pretrends_result" in result:
            pretrends = result["pretrends_result"]
            response += f"\nPre-trends Test\n\n"
            response += f"- Test Statistic: {pretrends.get('statistic', 'N/A')}\n"
            response += f"- p-value: {pretrends.get('pvalue', 'N/A')}\n"
            if pretrends.get("pvalue", 1) < 0.05:
                response += "- Result: Significant pre-trends detected\n"
            else:
                response += "- Result: No significant pre-trends\n"

        response += _format_pretrends_check(result)
        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in BJS Imputation estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_dcdh(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    cohort_col: Optional[str] = None,
    mode: str = "dyn",
    effects: int = 5,
    placebo: int = 5,
    controls: Optional[List[str]] = None,
) -> str:
    """
    de Chaisemartin & D'Haultfoeuille (2020) estimator.

    Uses the DIDmultiplegt approach to address negative weights problem
    in staggered DID designs with heterogeneous treatment effects.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        treatment_col: Treatment indicator variable name
        cohort_col: Treatment cohort variable (optional, auto-detected if None)
        mode: Estimation mode ("dyn" for dynamic, "avg" for average)
        effects: Number of dynamic effects to estimate (default: 5)
        placebo: Number of placebo periods to test (default: 5)
        controls: List of control variables (optional)

    Returns:
        Event study estimates with negative weights diagnostics

    Examples:
        >>> # Basic usage
        >>> estimate_dcdh("lemp", "countyreal", "year", "treat")

        >>> # With dynamic effects and placebos
        >>> estimate_dcdh("outcome", "unit", "time", "treatment", effects=7, placebo=2)

        >>> # With control variables
        >>> estimate_dcdh("y", "id", "t", "d", controls=["x1", "x2"])
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        # Use cohort_col if provided, otherwise use treatment_col
        gname_input = cohort_col if cohort_col else treatment_col
        processed = _preprocess_did_columns(unit_col, time_col, gname_input)
        actual_unit_col = processed["idname"]

        # Run DCDH estimation
        result = get_analyzer().r_estimators.dcdh_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            cohort_col=cohort_col,
            mode=mode,
            effects=effects,
            placebo=placebo,
            controls=controls,
        )

        if err := _check_result(result): return err

        _store_estimation("dcdh", result)

        # Format results
        response = "# de Chaisemartin & D'Haultfoeuille (2020) Results\n\n"
        response += f"Method: Fuzzy DID with Negative Weights Correction\n"
        response += f"Mode: {mode.upper()}\n"
        response += f"Effects Estimated: {effects} periods\n\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result)

        # DCDH-specific: negative weights diagnostics
        if "negative_weights" in result:
            neg_weights = result["negative_weights"]
            response += f"\nNegative Weights Diagnostics\n\n"
            response += f"- Share of Negative Weights: {neg_weights.get('share', 0):.1%}\n"
            response += f"- Min Weight: {neg_weights.get('min_weight', 'N/A')}\n"
            response += f"- Max Weight: {neg_weights.get('max_weight', 'N/A')}\n"
            if neg_weights.get("share", 0) > 0.1:
                response += "- Warning: Substantial negative weights detected!\n"
                response += "- Implication: TWFE estimates likely biased\n"
            else:
                response += "- Status: Low negative weights burden\n"

        # DCDH-specific: placebo tests
        if placebo > 0 and "placebo_tests" in result:
            placebo_results = result["placebo_tests"]
            response += f"\nPlacebo Tests (Pre-treatment)\n\n"
            response += "| Period | Estimate | Std. Error | p-value | Test Result |\n"
            response += "|--------|----------|------------|---------|-------------|\n"
            for p, test in placebo_results.items():
                test_result = "Pass" if test["pvalue"] > 0.05 else "Fail"
                response += (
                    f"| {p:^6} | {test['estimate']:^8.4f} | {test['se']:^10.4f} | "
                    f"{test['pvalue']:^7.4f} | {test_result:^11} |\n"
                )
            failed_tests = sum(
                1 for test in placebo_results.values() if test["pvalue"] <= 0.05
            )
            if failed_tests > 0:
                response += f"\nWarning: {failed_tests} placebo test(s) failed\n"
                response += "This suggests potential violations of parallel trends.\n"
            else:
                response += "\nAll placebo tests passed\n"

        # Method-specific insights
        response += "\nMethod Insights\n\n"
        response += "- Advantage: Addresses negative weights problem in TWFE\n"
        response += "- Innovation: Explicitly models treatment effect heterogeneity\n"
        response += "- Robustness: Built-in placebo testing capability\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in DCDH estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_efficient(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,
    estimand: str = "simple",
    event_time: Optional[List[int]] = None,
    use_cs: bool = False,
    use_sa: bool = False,
    beta: Optional[float] = 1.0,
) -> str:
    """
    Roth & Sant'Anna (2023) efficient estimator.

    Uses the staggered package for statistically efficient estimation
    in staggered adoption designs with heterogeneous treatment effects.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        cohort_col: Treatment cohort variable (first treatment time)
        estimand: Estimand type ("simple", "cohort", "calendar")
        event_time: List of relative time periods to estimate (optional)
        use_cs: Use Callaway & Sant'Anna method for comparison (sets beta=1, use_last_treated_only=False)
        use_sa: Use Sun & Abraham method for comparison (sets beta=1, use_last_treated_only=True)
        beta: Efficiency parameter. None (default) = optimal plug-in efficient beta.
              beta=0: Simple difference-in-means
              beta=1: Callaway & Sant'Anna weighting
              Ignored if use_cs or use_sa is True.

    Returns:
        Event study estimates with efficient ATT

    Examples:
        >>> # Basic efficient estimation
        >>> estimate_efficient("lemp", "countyreal", "year", "first.treat")

        >>> # With specific event times
        >>> estimate_efficient("outcome", "unit", "time", "cohort",
        ...                    event_time=[-2, -1, 0, 1, 2])

        >>> # With method comparison
        >>> estimate_efficient("y", "id", "t", "g", use_cs=True, use_sa=True)
    """
    # This estimator is disabled due to systematic numerical issues across
    # multiple benchmark datasets (see KNOWN_ISSUES.md #1).
    # Use estimate_callaway_santanna, estimate_bjs_imputation, or
    # estimate_gardner_two_stage instead.
    return (
        "# Efficient Estimator (Roth & Sant'Anna 2023) - Temporarily Disabled\n\n"
        "This method produced unreliable results across benchmark datasets "
        "(magnitude errors, direction reversals, and runtime failures).\n\n"
        "Use these alternatives instead:\n"
        "- estimate_callaway_santanna - Doubly robust, widely trusted\n"
        "- estimate_bjs_imputation - Fast imputation-based estimator\n"
        "- estimate_gardner_two_stage - Computationally simple two-stage\n"
    )


@mcp.tool()
async def estimate_gardner_two_stage(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,  # REQUIRED - not Optional
    covariates: Optional[List[str]] = None,
) -> str:
    """
    Gardner (2022) two-stage DID estimator with EVENT STUDY specification.

    Uses the did2s package for two-stage estimation that is robust to
    heterogeneous treatment effects. ALWAYS uses event study specification
    because static treatment effects are not identified with unit+time FE.

    IMPORTANT: Gardner's method REQUIRES cohort variable for event study.
    Static treatment indicators cause collinearity with fixed effects.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        cohort_col: Cohort/group variable indicating FIRST TREATMENT TIME
                   (e.g., 'first.treat' with values 0, 2004, 2006, 2007)
                   REQUIRED for proper identification
        covariates: List of control variables (optional)

    Returns:
        Event study estimates from two-stage approach

    Examples:
        >>> # Correct usage with cohort variable (EVENT STUDY)
        >>> estimate_gardner_two_stage("lemp", "countyreal", "year", "first.treat")

        >>> # With control variables
        >>> estimate_gardner_two_stage("y", "id", "t", "cohort", covariates=["x1", "x2"])

    Note:
        For simple ATT without event study, use Callaway & Sant'Anna or
        Efficient estimator instead.

    Reference:
        Gardner (2022) "Two-stage differences in differences"
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, cohort_col)
        actual_unit_col = processed["idname"]
        actual_cohort_col = processed["gname"]

        # Run Gardner Two-Stage estimation (EVENT STUDY mode always)
        result = get_analyzer().r_estimators.gardner_two_stage_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            cohort_col=actual_cohort_col,
            covariates=covariates,
        )

        if err := _check_result(result): return err

        _store_estimation("gardner_two_stage", result)

        # Format results
        response = "# Gardner (2022) Two-Stage DID Results\n\n"
        response += "Method: Two-Stage DID Estimator\n"
        if covariates:
            response += f"Covariates: {len(covariates)} variables\n"
        response += "\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result)

        # Gardner-specific: two-stage procedure details
        if "stage_info" in result:
            stage_info = result["stage_info"]
            response += "\nTwo-Stage Procedure Details\n\n"
            response += "- Stage 1: Impute counterfactual outcomes using never-treated units\n"
            response += "- Stage 2: Estimate treatment effects using imputed outcomes\n"
            if "imputation_r2" in stage_info:
                response += f"- Imputation R²: {stage_info['imputation_r2']:.3f}\n"
            if "treated_units" in stage_info:
                response += f"- Treated Units: {stage_info['treated_units']:,}\n"
            if "control_units" in stage_info:
                response += f"- Control Units: {stage_info['control_units']:,}\n"

        response += _format_pretrends_check(result)

        # Method-specific insights
        response += "\nMethod Insights\n\n"
        response += "- Advantage: Computationally simple and fast\n"
        response += "- Innovation: Two-stage imputation approach\n"
        response += "- Robustness: Handles heterogeneous treatment effects\n"
        response += "- Flexibility: Easy to incorporate covariates\n"
        response += "- Implementation: Built on standard regression methods\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in Gardner Two-Stage estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_gsynth(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    covariates: Optional[List[str]] = None,
    force: str = "two-way",
    CV: bool = True,
    r_range: tuple = (0, 5),
    se: bool = True,
    inference: str = "parametric",
) -> str:
    """
    Generalized Synthetic Control Method (Xu 2017).

    Uses interactive fixed effects model to estimate treatment effects when
    parallel trends may be violated due to unobserved time-varying confounders.
    Suitable for staggered adoption designs.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        treatment_col: Binary treatment indicator (0/1)
        covariates: List of time-varying covariate names (optional)
        force: Fixed effects - "none", "unit", "time", or "two-way" (default: "two-way")
        CV: Use cross-validation to select number of factors (default: True)
        r_range: Tuple (min, max) for number of factors to consider (default: (0, 5))
        se: Compute standard errors (default: True)
        inference: "parametric" or "nonparametric" (bootstrap) (default: "parametric")

    Returns:
        Formatted estimation results with ATT and diagnostics

    Reference:
        Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference
        with Interactive Fixed Effects Models." Political Analysis, 25(1), 57-76.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, treatment_col)
        actual_unit_col = processed["idname"]

        # Run gsynth estimation
        result = get_analyzer().r_estimators.gsynth_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            covariates=covariates,
            force=force,
            CV=CV,
            r_range=r_range,
            se=se,
            inference=inference,
        )

        if err := _check_result(result): return err

        _store_estimation("gsynth", result)

        # Format results
        response = "# Generalized Synthetic Control (Xu 2017) Results\n\n"
        response += f"Method: Interactive Fixed Effects Model\n"
        response += f"Fixed Effects: {result['force']}\n"
        response += f"Inference: {result['inference']}\n"
        if covariates:
            response += f"Covariates: {len(covariates)} variables\n"
        response += "\n"
        response += _format_overall_att(result)

        # gsynth-specific: model information
        response += "Model Information\n\n"
        response += f"- Number of Factors: {result['n_factors']}\n"
        response += f"- Treated Units: {result['n_treated']}\n"
        response += f"- Control Units: {result['n_control']}\n"
        response += f"- Time Periods: {result['n_periods']}\n"
        if "pre_treatment_mspe" in result:
            response += f"- Pre-treatment MSPE: {result['pre_treatment_mspe']:.4f}\n"
        response += "\n"

        # Method insights
        response += "Method Insights\n\n"
        response += "- Advantage: Relaxes parallel trends assumption via interactive fixed effects\n"
        response += "- Innovation: Estimates latent factors capturing time-varying confounders\n"
        response += "- Robustness: Handles staggered adoption automatically\n"
        response += "- Flexibility: Cross-validation selects optimal number of factors\n"
        response += "- Best Use: When parallel trends violated due to unobserved time-varying confounders\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in gsynth estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_synthdid(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    cohort_col: Optional[str] = None,
    vcov_method: str = "placebo",
) -> str:
    """
    Synthetic Difference-in-Differences estimator (Arkhangelsky et al. 2019).

    Combines synthetic control method with difference-in-differences by
    estimating both unit and time weights to construct a synthetic control.

    IMPORTANT: Requires all treated units to begin treatment simultaneously.
    For staggered adoption, use gsynth or other DID estimators instead.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        treatment_col: Binary treatment indicator (0/1)
        cohort_col: Optional cohort variable (for checking staggered adoption)
        vcov_method: Variance estimation - "placebo", "bootstrap", or "jackknife"

    Returns:
        Formatted estimation results with ATT, comparison methods, and weights

    Reference:
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2019). "Synthetic Difference in Differences." NBER Working Paper 25532.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        gname_input = cohort_col if cohort_col else treatment_col
        processed = _preprocess_did_columns(unit_col, time_col, gname_input)
        actual_unit_col = processed["idname"]

        # Run synthdid estimation
        result = get_analyzer().r_estimators.synthdid_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            cohort_col=cohort_col,
            vcov_method=vcov_method,
        )

        if err := _check_result(result): return err

        _store_estimation("synthdid", result)

        # Format results
        response = "# Synthetic Difference-in-Differences (Arkhangelsky et al. 2019) Results\n\n"
        response += f"Method: Synthetic DiD\n"
        response += f"Variance Estimation: {result['vcov_method']}\n\n"
        response += _format_overall_att(result)

        # synthdid-specific: method comparison
        if "comparison_methods" in result:
            comp = result["comparison_methods"]
            response += "Method Comparison\n\n"
            response += "| Method | Estimate | Description |\n"
            response += "|--------|----------|-------------|\n"
            response += f"| Traditional DiD | {comp['traditional_did']['estimate']:.4f} | {comp['traditional_did']['note']} |\n"
            response += f"| Synthetic Control | {comp['synthetic_control']['estimate']:.4f} | {comp['synthetic_control']['note']} |\n"
            response += f"| Synthetic DiD | {comp['synthdid']['estimate']:.4f} | {comp['synthdid']['note']} |\n\n"

        # synthdid-specific: sample information
        response += "Sample Information\n\n"
        response += f"- Treated Units: {result['n_treated_units']}\n"
        response += f"- Control Units: {result['n_control_units']}\n"
        response += f"- Pre-treatment Periods: {result['n_pretreatment_periods']}\n"
        response += f"- Post-treatment Periods: {result['n_posttreatment_periods']}\n"

        # synthdid-specific: weights
        if "unit_weights" in result:
            response += f"\nUnit Weights\n"
            response += f"- Non-zero weights: {result['unit_weights']['n_nonzero']} control units\n"
            response += f"- Maximum weight: {result['unit_weights']['max_weight']:.4f}\n"
        if "time_weights" in result:
            response += f"\nTime Weights\n"
            response += f"- Non-zero weights: {result['time_weights']['n_nonzero']} pre-treatment periods\n"
            response += f"- Maximum weight: {result['time_weights']['max_weight']:.4f}\n"
        response += "\n"

        # Method insights
        response += "Method Insights\n\n"
        response += "- Advantage: Robust to both parallel trends and interpolation bias\n"
        response += "- Innovation: Combines synthetic control unit weighting with DiD time weighting\n"
        response += "- Robustness: Estimates both unit weights (which controls) and time weights (which periods)\n"
        response += "- Comparison: Automatically compares with traditional DiD and SC methods\n"
        response += "- Best Use: Simultaneous treatment timing with potential violations of parallel trends\n"
        response += "- Limitation: Requires all treated units to start treatment at same time\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in synthdid estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_drdid(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    covariates: Optional[List[str]] = None,
    panel: bool = True,
    est_method: str = "imp",
    boot: bool = False,
    nboot: int = 999,
) -> str:
    """
    Doubly Robust Difference-in-Differences estimator (Sant'Anna & Zhao 2020).

    Provides doubly robust estimation combining outcome regression and inverse
    probability weighting. Consistent if either the outcome model or the
    propensity score model is correctly specified.

    Best suited for canonical 2-period DID designs with covariates.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        treatment_col: Binary treatment indicator (0/1)
        covariates: List of covariate names for outcome and propensity models (optional)
        panel: True for panel data, False for repeated cross-sections (default: True)
        est_method: Estimation method:
            - "imp" (default): Improved doubly robust (Sant'Anna & Zhao 2020)
            - "trad": Traditional doubly robust
            - "ipw": Inverse probability weighting only
            - "reg": Outcome regression only
        boot: Use bootstrap for inference (default: False uses analytical)
        nboot: Number of bootstrap replications (default: 999)

    Returns:
        Formatted estimation results with ATT

    Reference:
        Sant'Anna, P. H. C. & Zhao, J. (2020). "Doubly Robust Difference-in-
        Differences Estimators." Journal of Econometrics, 219(1), 101-122.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, treatment_col)
        actual_unit_col = processed["idname"]

        # Run DRDID estimation
        result = get_analyzer().r_estimators.drdid_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            covariates=covariates,
            panel=panel,
            est_method=est_method,
            boot=boot,
            nboot=nboot,
        )

        if err := _check_result(result): return err

        _store_estimation("drdid", result)

        # Format results
        response = "# Doubly Robust DID (Sant'Anna & Zhao 2020) Results\n\n"
        response += f"Method: {result['method']}\n"
        response += f"Estimation: {est_method.upper()}\n"
        response += f"Data Type: {'Panel' if panel else 'Repeated Cross-Sections'}\n"
        if covariates:
            response += f"Covariates: {', '.join(covariates)}\n"
        response += "\n"
        response += _format_overall_att(result, label="Average Treatment Effect on the Treated (ATT)")

        # Method insights
        response += "Method Insights\n\n"
        response += "- Double Robustness: Consistent if either outcome OR propensity model is correct\n"
        response += "- Improved Estimator: Uses locally efficient semiparametric estimation\n"
        response += "- Best Use: Canonical 2-period DID with covariates\n"
        response += "- Limitation: Designed for 2-period settings; for staggered adoption, use CS or etwfe\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in DRDID estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def estimate_etwfe(
    outcome_col: str,
    unit_col: str,
    time_col: str,
    cohort_col: str,
    treatment_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    vcov: str = "by",
    by_cohort: bool = True,
) -> str:
    """
    Extended TWFE estimator (Wooldridge 2021, 2023).

    Implements Wooldridge's extended TWFE approach that includes cohort-time
    interactions to correctly handle heterogeneous treatment effects in
    staggered DID designs. Built on fixest for fast estimation.

    Args:
        outcome_col: Outcome variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        cohort_col: Cohort variable (first treatment time; 0 for never-treated)
        treatment_col: Binary treatment indicator (optional, derived from cohort if absent)
        covariates: List of covariate names (optional)
        vcov: Variance-covariance type:
            - "by" (default): Cluster by cohort*period groups
            - unit column name: Cluster by unit
            - "hetero": Heteroskedasticity-robust
        by_cohort: Report effects by cohort (default: True)

    Returns:
        Formatted estimation results with ATT and event study

    Reference:
        Wooldridge, J. M. (2021). "Two-Way Fixed Effects, the Two-Way
        Mundlak Regression, and Difference-in-Differences Estimators."
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, cohort_col)
        actual_unit_col = processed["idname"]
        actual_cohort_col = processed["gname"]

        # Run etwfe estimation
        result = get_analyzer().r_estimators.etwfe_estimator(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            cohort_col=actual_cohort_col,
            treatment_col=treatment_col,
            covariates=covariates,
            vcov=vcov,
            by_cohort=by_cohort,
        )

        if err := _check_result(result): return err

        _store_estimation("etwfe", result)

        # Format results
        response = "# Extended TWFE (Wooldridge 2021) Results\n\n"
        response += f"Method: {result['method']}\n"
        response += f"VCOV Type: {result['vcov_type']}\n"
        if covariates:
            response += f"Covariates: {', '.join(covariates)}\n"
        response += "\n"
        response += _format_overall_att(result)
        response += _format_event_study_table(result, show_ci=False)
        response += _format_pretrends_check(result)

        # etwfe-specific: cohort effects
        if result.get("cohort_effects"):
            response += "Cohort-Specific Effects\n\n"
            response += "| Cohort | Estimate | Std. Error |\n"
            response += "|--------|----------|------------|\n"
            for cohort, eff in result["cohort_effects"].items():
                response += f"| {cohort} | {eff['estimate']:.4f} | {eff['se']:.4f} |\n"
            response += "\n"

        # Method insights
        response += "Method Insights\n\n"
        response += "- Approach: Saturated model with cohort-time interactions\n"
        response += "- Advantage: Correctly handles heterogeneous treatment effects in TWFE framework\n"
        response += "- Innovation: Wooldridge shows that augmented TWFE with proper interactions is valid\n"
        response += "- Robustness: Uses marginaleffects for proper ATT aggregation\n"
        response += "- Best Use: Staggered DID with covariates where TWFE framework is preferred\n"

        response += _format_significance(result)

        return response

    except Exception as e:
        logger.error(f"Error in etwfe estimation: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def create_panel_view(
    outcome_col: str,
    treatment_col: str,
    unit_col: str,
    time_col: str,
    output_path: Optional[str] = None,
    view_type: str = "treat",
    by_timing: bool = True,
) -> str:
    """
    Create treatment status visualization using the R panelView package.

    Produces a heatmap showing treatment timing across units, essential for
    understanding staggered DID designs before estimation.

    Args:
        outcome_col: Outcome variable name
        treatment_col: Treatment variable name
        unit_col: Unit identifier name
        time_col: Time variable name
        output_path: Path to save the plot (PNG). If None, saves to temp file.
        view_type: Type of visualization:
            - "treat" (default): Treatment status heatmap
            - "outcome": Outcome trajectories
            - "bivariate": Combined treatment and outcome view
        by_timing: Sort units by treatment timing (default: True)

    Returns:
        Summary of visualization with plot path

    Reference:
        Mou, H., Liu, L. & Xu, Y. (2023). "Panel Data Visualization in R
        (panelView) and Stata (panelview)." Journal of Statistical Software.
    """
    try:
        if err := _require_data(): return err
        if err := _require_r(): return err

        # Auto-preprocess data (same logic as workflow)
        processed = _preprocess_did_columns(unit_col, time_col, treatment_col)
        actual_unit_col = processed["idname"]

        # Create panelView visualization
        result = get_analyzer().r_estimators.create_panel_view(
            data=get_analyzer().data,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            unit_col=actual_unit_col,
            time_col=time_col,
            output_path=output_path,
            view_type=view_type,
            by_timing=by_timing,
        )

        if err := _check_result(result): return err

        # Format results
        response = "# Panel Data Visualization (panelView)\n\n"
        response += f"View Type: {result['view_type']}\n"
        response += f"Plot Saved To: {result['plot_path']}\n\n"

        response += "Data Summary\n\n"
        response += f"- Units: {result['n_units']}\n"
        response += f"- Time Periods: {result['n_periods']}\n\n"

        if result.get("treatment_groups"):
            response += "Treatment Timing Groups\n\n"
            response += "| Treatment Time | Number of Units |\n"
            response += "|---------------|----------------|\n"
            for timing, count in sorted(
                result["treatment_groups"].items(), key=lambda x: str(x[0])
            ):
                label = (
                    "Never Treated"
                    if timing == "never_treated"
                    else f"First treated: {timing}"
                )
                response += f"| {label} | {count} |\n"
            response += "\n"

        response += "Usage Notes\n\n"
        response += "- The heatmap shows treatment status (treated=dark, untreated=light) across units and time\n"
        response += "- Units are sorted by treatment timing when by_timing=True\n"
        response += "- Use view_type='outcome' to visualize outcome trajectories\n"
        response += (
            "- Use view_type='bivariate' for combined treatment and outcome view\n"
        )

        return response

    except Exception as e:
        logger.error(f"Error in panelView: {e}")
        return f"Error: {str(e)}"


# =============================================================================
# VISUALIZATION TOOLS
# =============================================================================


@mcp.tool()
async def create_event_study_plot(
    results_key: str = "latest",
    backend: str = "matplotlib",
    display_inline: bool = True,
    save_path: str = None,
    auto_optimize: bool = True,
):
    """
    Create event study plot from DID estimation results.

    Args:
        results_key: Which results to plot. Options:
            - "latest" - Most recently run method (default)
            - "callaway_santanna" - Callaway & Sant'Anna (2021)
            - "sun_abraham" - Sun & Abraham (2021)
            - "bjs_imputation" - Borusyak, Jaravel & Spiess (2024)
            - "dcdh" - de Chaisemartin & D'Haultfoeuille (2020)
            - "gardner_two_stage" - Gardner (2022) Two-Stage
            - "gsynth" - Generalized Synthetic Control (Xu 2017)
            - "synthdid" - Synthetic DID (Arkhangelsky et al. 2019)
            - "efficient" - DISABLED (see KNOWN_ISSUES.md)
        backend: Visualization backend ("matplotlib" or "plotly")
        display_inline: Whether to display image inline in Claude Desktop (default: True)
        save_path: Optional filename to save the plot
        auto_optimize: Automatically compress image if needed (default: True)

    Returns:
        If display_inline=True: Image content for display + text description
        If display_inline=False: Text description with file path only

    Examples:
        >>> # Plot most recent results
        >>> create_event_study_plot()

        >>> # Plot specific method
        >>> create_event_study_plot(results_key="callaway_santanna")

        >>> # Save without displaying
        >>> create_event_study_plot(results_key="sun_abraham",
        ...                         display_inline=False,
        ...                         save_path="sa_plot.png")
    """
    try:
        # Import MCP types at runtime to avoid startup conflicts
        # Determine display mode based on parameter
        display_mode = "both" if display_inline else "save"

        result = await get_analyzer().create_event_study_plot(
            results_key=results_key,
            backend=backend,
            save_path=save_path,
            display_mode=display_mode,
            auto_optimize=auto_optimize,
        )

        # Handle error case
        if isinstance(result, dict) and result.get("status") == "error":
            return f"Error creating event study plot: {result['message']}"

        # If display_mode was "display", result is ImageContent directly
        # Check using type name to avoid import issues
        if hasattr(result, "__class__") and result.__class__.__name__ == "ImageContent":
            return result

        # Otherwise result is a dict with metadata
        if not isinstance(result, dict):
            return f"Unexpected result type: {type(result)}"

        # Create text description
        text_desc = f"""
# Event Study Plot Created

Method: {result.get("method", "Unknown")}
Backend: {result["backend"]}
File: {result["file_path"]}

The event study plot has been generated and saved. The plot shows:
- Point estimates for each event time
- 95% confidence intervals
- Pre-treatment period highlighting
- Treatment start indicator
"""

        # Add optimization info if applicable
        if result.get("optimized", False):
            text_desc += f"\n*Image was optimized for display (size: {result.get('file_size', 'unknown')} bytes)*"

        # If display_inline and image_content available, return both text and image
        if display_inline and "image_content" in result:
            if TextContent is not None:
                return [
                    TextContent(type="text", text=text_desc),
                    result["image_content"],
                ]
            else:
                # Fallback: return text + warning
                return (
                    text_desc + "\n\nMCP types not available, image display disabled."
                )

        # If display_inline but image too large
        if display_inline and not result.get("can_display_inline", True):
            text_desc += "\n\nImage size exceeds 1MB limit. Saved to file only."

        # Return text description only
        return text_desc

    except Exception as e:
        logger.error(f"Error in create_event_study_plot tool: {e}", exc_info=True)
        return f"Error: {str(e)}"


@mcp.tool()
async def debug_analyzer_state() -> str:
    """
    Debug tool to inspect current analyzer state.

    Returns detailed information about:
    - Analyzer instance ID
    - Available diagnostic results
    - Available estimation results
    - Data loading status
    """
    analyzer_instance = get_analyzer()

    response = "# Analyzer State Debug Report\n\n"
    response += f"Analyzer Instance ID: {id(analyzer_instance)}\n\n"

    response += "Data Status\n"
    if analyzer_instance.data is not None:
        response += f"- Data loaded: {analyzer_instance.data.shape[0]} rows × {analyzer_instance.data.shape[1]} columns\n\n"
    else:
        response += "- No data loaded\n\n"

    response += "Diagnostics Storage\n"
    if analyzer_instance.diagnostics:
        response += (
            f"- Keys available: {list(analyzer_instance.diagnostics.keys())}\n"
        )
        for key, value in analyzer_instance.diagnostics.items():
            if isinstance(value, dict):
                response += f"- {key}: Dict with keys {list(value.keys())[:5]}...\n"
            else:
                response += f"- {key}: {type(value).__name__}\n"
    else:
        response += "- No diagnostics stored (empty dict)\n"
    response += "\n"

    response += "Results Storage\n"
    if analyzer_instance.results:
        response += f"- Keys available: {list(analyzer_instance.results.keys())}\n"
    else:
        response += "- No results stored (empty dict)\n"

    return response


@mcp.tool()
async def create_diagnostic_plots(
    backend: str = "matplotlib", save_path: str = None
) -> str:
    """
    Create diagnostic plots for TWFE bias analysis.

    IMPORTANT - PREREQUISITES REQUIRED 

    This tool REQUIRES diagnostic analysis to be run FIRST. You MUST run at least
    one of the following diagnostic tools before calling this function:

    Required Steps (run BEFORE this tool):
    1. diagnose_goodman_bacon() - For Goodman-Bacon decomposition plots
       - Required parameters: formula, id_var, time_var
       - Example: diagnose_goodman_bacon(formula="lemp ~ treat", id_var="countyreal", time_var="year")

    2. analyze_twfe_weights() - For TWFE negative weights plots
       - Required parameters: outcome_col, unit_col, time_col, treatment_col
       - Example: analyze_twfe_weights(outcome_col="lemp", unit_col="countyreal", time_col="year", treatment_col="treat")

    Complete Workflow:
    1. Load data: load_data("path/to/data.csv")
    2. Run diagnostics (one or both):
       - diagnose_goodman_bacon(formula="outcome ~ treatment", id_var="unit_id", time_var="time_id")
       - analyze_twfe_weights(outcome_col="outcome", unit_col="unit_id", time_col="time_id", treatment_col="treatment")
    3. Create plots: create_diagnostic_plots() ← You are here

    Troubleshooting:
    - If you get "No diagnostic results available", use debug_analyzer_state() to check what's stored
    - Make sure you didn't restart the MCP server between running diagnostics and creating plots

    Args:
        backend: Visualization backend ("matplotlib" or "plotly")
        save_path: Optional filename prefix for saving plots

    Returns:
        Success message with plot details or error message if prerequisites not met
    """
    try:
        analyzer_instance = get_analyzer()
        logger.info(
            f"create_diagnostic_plots called. Analyzer instance ID: {id(analyzer_instance)}"
        )
        logger.info(
            f"Available diagnostics keys: {list(analyzer_instance.diagnostics.keys())}"
        )
        logger.info(f"Diagnostics content: {analyzer_instance.diagnostics}")

        result = await analyzer_instance.create_diagnostic_plots(
            backend=backend, save_path=save_path
        )

        if result["status"] == "success":
            plots_info = []
            for plot_type, plot_data in result["plots"].items():
                if plot_data["status"] == "success":
                    plots_info.append(
                        f"- {plot_type.replace('_', ' ').title()}: {plot_data['file_path']}"
                    )

            return f"""
# Diagnostic Plots Created

Backend: {backend}
Number of plots: {result["n_plots"]}

Generated plots:
{chr(10).join(plots_info)}

These diagnostic plots help identify potential TWFE bias:
- Goodman-Bacon Decomposition: Shows weight distribution across comparison types
- TWFE Weights Analysis: Identifies negative weights and their impact

All plots have been saved to the figures directory.
"""
        else:
            # Enhanced error message with actionable suggestions
            error_msg = result["message"]

            # Check what's available
            available_diag = list(analyzer_instance.diagnostics.keys())

            return f"""Error: Cannot create diagnostic plots - {error_msg}

Current Status:
- Available diagnostic results: {available_diag if available_diag else "None"}
- You need at least one diagnostic analysis to create plots

REQUIRED: Run diagnostic analysis FIRST

Step-by-Step Instructions:

1️⃣ Check current state (optional but recommended):
   
   debug_analyzer_state()
   

2️⃣ Run diagnostic analysis (choose at least one):

   Option A - Goodman-Bacon Decomposition:
   
   diagnose_goodman_bacon(
       formula="lemp ~ treat",        # Replace with your outcome ~ treatment
       id_var="countyreal",            # Replace with your unit ID column
       time_var="year"                 # Replace with your time column
   )
   

   Option B - TWFE Weights Analysis:
   
   analyze_twfe_weights(
       outcome_col="lemp",             # Replace with your outcome variable
       unit_col="countyreal",          # Replace with your unit ID column
       time_col="year",                # Replace with your time column
       treatment_col="treat"           # Replace with your treatment variable
   )
   

3️⃣ Then retry:
   
   create_diagnostic_plots()
   

Tip: Run BOTH diagnostics for a comprehensive analysis!
"""

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def create_comprehensive_report(
    results_key: str = "latest", backend: str = "matplotlib", save_path: str = None
) -> str:
    """
    Create comprehensive DID analysis report with all visualizations.

    Args:
        results_key: Which results to include ("latest" or specific key)
        backend: Visualization backend ("matplotlib" or "plotly")
        save_path: Optional filename for the report
    """
    try:
        result = await get_analyzer().create_comprehensive_report(
            results_key=results_key, backend=backend, save_path=save_path
        )

        if result["status"] == "success":
            return f"""
# Comprehensive DID Report Created

Report Type: {result["report_type"]}
Number of plots: {result["n_plots"]}
Report file: {result["report_path"]}

The comprehensive report includes:
- Executive Summary: Key findings and recommendations
- Estimation Results: Detailed results from your chosen method
- Diagnostic Analysis: TWFE bias assessment
- Event Study Plot: Treatment effects over time
- Diagnostic Plots: Goodman-Bacon decomposition and weights analysis

This report is ready for presentation to stakeholders or inclusion in research papers.
The HTML report can be opened in any web browser for interactive viewing.
"""
        else:
            return f"Error creating comprehensive report: {result['message']}"

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_visualization_info() -> str:
    """Get information about available visualization backends and capabilities."""
    try:
        backends = get_analyzer().get_visualization_backends()

        return f"""
# Visualization Capabilities

Available backends:
{chr(10).join([f"- {backend}" for backend in backends])}

Supported plot types:
- Event Study Plots: Treatment effects over time with confidence intervals
- Goodman-Bacon Decomposition: Weight distribution across comparison types
- TWFE Weights Analysis: Negative weights detection and visualization
- Comprehensive Reports: Combined HTML reports with all plots

Output formats:
- PNG images (high resolution)
- HTML interactive plots (Plotly)
- Base64 encoded images (for embedding)
- Comprehensive HTML reports

Usage workflow:
1. Load data: load_data with your dataset
2. Run estimation: estimate_callaway_santanna or other methods
3. Run diagnostics: diagnose_twfe for bias analysis
4. Create plots: Use visualization tools to generate publication-ready figures
5. Generate reports: Create comprehensive reports for presentations

All plots are automatically saved to the figures/ directory.
"""

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def manage_storage(
    action: str = "stats", cleanup_old: bool = False, enforce_limits: bool = False
) -> str:
    """
    Manage output file storage.

    Args:
        action: Action to perform ("stats", "cleanup", "list")
        cleanup_old: Remove files older than retention period
        enforce_limits: Enforce maximum file count limits

    Returns:
        Storage management report
    """
    try:
        storage = get_storage()

        if action == "stats":
            stats = storage.get_storage_stats()
            return f"""
# Storage Statistics

Base Directory: {stats["base_directory"]}
Total Files: {stats["total_files"]}
Total Size: {stats["total_size_formatted"]}
 By Type:
{
                chr(10).join(
                    [
                        f"- {k}: {v['count']} files ({storage._format_size(v['size'])})"
                        for k, v in stats["by_type"].items()
                    ]
                )
            }
 Management Options:
- Use cleanup_old=True to remove files older than {storage.cleanup_after_days} days
- Use enforce_limits=True to limit to {storage.max_files_per_type} files per type
"""

        elif action == "cleanup":
            removed_count = 0
            if cleanup_old:
                removed_count += storage.cleanup_old_files()
            if enforce_limits:
                removed_count += storage.enforce_storage_limits()

            return f"""
# Storage Cleanup Complete

Files Removed: {removed_count}

The storage has been cleaned according to your settings.
Run with action="stats" to see the current storage status.
"""

        elif action == "list":
            resources = storage.list_resources()[:10]  # Show first 10
            return f"""
# Recent Output Files

{
                chr(10).join(
                    [
                        f"- {r['name']} ({r['mimeType']}): {storage._format_size(r.get('size', 0))}"
                        for r in resources
                    ]
                )
            }

Total available: {len(storage.list_resources())} files

Use MCP resource listing to access all files programmatically.
"""

        else:
            return f"Unknown action: {action}. Use 'stats', 'cleanup', or 'list'"

    except Exception as e:
        logger.error(f"Error in storage management: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def export_results(
    format: str = "csv",
    results_key: str = "latest",
    filename: Optional[str] = None,
    include_diagnostics: bool = False,
) -> str:
    """
    Export DID analysis results to various formats.

    Supports multiple export formats for different use cases:
    - CSV: For data analysis in Excel/R/Stata
    - Excel: For formatted reports with multiple sheets
    - LaTeX: For academic papers (publication-ready tables)
    - Markdown: For documentation and reports
    - JSON: For programmatic access and archival

    Args:
        format: Export format ("csv", "excel", "latex", "markdown", "json")
        results_key: Which results to export. Options:
            - "latest" - Most recently run method (default, recommended)

            Naming Convention:
            - Methods called directly: Use method name without prefix
              - "callaway_santanna" - Callaway & Sant'Anna (2021)
              - "sun_abraham" - Sun & Abraham (2021)
              - "bjs_imputation" - Borusyak, Jaravel & Spiess (2024)
              - "dcdh" - de Chaisemartin & D'Haultfoeuille (2020)
              - "gardner_two_stage" - Gardner (2022) Two-Stage
              - "gsynth" - Generalized Synthetic Control (Xu 2017)
              - "synthdid" - Synthetic DID (Arkhangelsky et al. 2019)
              - "efficient" - DISABLED (see KNOWN_ISSUES.md)

            - Methods called via workflow(): Use "workflow_" prefix
              - "workflow_callaway_santanna"
              - "workflow_sun_abraham"
              - "workflow_imputation_bjs"
              - etc.

            Tip: Use "latest" to export the primary estimation results.

        filename: Custom filename (optional, auto-generated if not provided)
        include_diagnostics: Whether to include diagnostic information (default: False)

    Returns:
        Export confirmation with file path and next steps

    Examples:
        >>> # Export latest results (recommended)
        >>> export_results("csv")

        >>> # Export results from direct method call
        >>> export_results("latex", "callaway_santanna", "my_results.tex")

        >>> # Export results from workflow call
        >>> export_results("markdown", "workflow_callaway_santanna")

        >>> # Export with diagnostics
        >>> export_results("markdown", "sun_abraham", include_diagnostics=True)
    """
    try:
        analyzer = get_analyzer()

        # Get results to export
        if results_key == "latest":
            results = analyzer.results.get("latest", {})
            if not results:
                # Try to find most recent result
                for key in reversed(list(analyzer.results.keys())):
                    if key not in ["diagnostics", "workflow"]:
                        results = analyzer.results[key]
                        results_key = key
                        break
        else:
            results = analyzer.results.get(results_key, {})

        if not results:
            # Provide helpful error with available keys
            available_keys = [
                k
                for k in analyzer.results.keys()
                if k not in ["latest", "diagnostics", "workflow"]
            ]
            if not available_keys:
                return "Error: No results found to export. Please run a DID estimation method first."
            else:
                keys_str = "', '".join(available_keys)
                if results_key == "latest":
                    return f"Error: No results available. Please run a DID estimation method first."
                else:
                    return f"""Error: Results key '{results_key}' not found.

Available results keys:
- '{keys_str}'

Tip: Use results_key="latest" for the primary estimation results, or specify one of the available keys above.
"""

        # Include diagnostics if requested
        export_data = {"results": results}
        if include_diagnostics and analyzer.diagnostics:
            export_data["diagnostics"] = analyzer.diagnostics

        # Format content based on export format
        if format == "csv":
            content = _format_results_as_csv(export_data)
            extension = "csv"
        elif format == "excel":
            content = _format_results_as_excel(export_data)
            extension = "xlsx"
        elif format == "latex":
            content = _format_results_as_latex(export_data)
            extension = "tex"
        elif format == "markdown":
            content = _format_results_as_markdown(export_data)
            extension = "md"
        elif format == "json":
            content = json.dumps(export_data, indent=2, default=str)
            extension = "json"
        else:
            return f"Error: Unsupported format '{format}'. Use: csv, excel, latex, markdown, or json"

        # Prepare filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"did_results_{results_key}_{timestamp}.{extension}"
        elif not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"

        # Save file using storage manager
        storage = get_storage()
        file_path = storage.base_dir / "exports" / filename

        # Ensure directory exists and resolve to absolute path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = file_path.resolve()  # Convert to absolute path

        logger.info(f"Exporting results to: {file_path}")
        logger.info(f"Storage base directory: {storage.base_dir.resolve()}")

        # Write content
        if format == "excel":
            # Excel requires special handling
            import pandas as pd

            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                # Write main results
                if "event_study" in results:
                    df = pd.DataFrame(results["event_study"]).T
                    df.to_excel(writer, sheet_name="Event Study")
                if "overall_att" in results:
                    df = pd.DataFrame([results["overall_att"]])
                    df.to_excel(writer, sheet_name="Overall ATT")
                if include_diagnostics and "diagnostics" in export_data:
                    # Add diagnostics sheet
                    diag_data = export_data["diagnostics"]
                    if isinstance(diag_data, dict):
                        df = pd.DataFrame([diag_data])
                        df.to_excel(writer, sheet_name="Diagnostics")
        else:
            # Text-based formats
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Verify file was written successfully
        if not file_path.exists():
            raise FileNotFoundError(
                f"File write reported success but file not found: {file_path}"
            )

        file_size = file_path.stat().st_size
        logger.info(f"File written successfully: {file_path} ({file_size:,} bytes)")

        # Generate success message
        response = f"Results Exported Successfully\n\n"
        response += f"File: {file_path}\n"
        response += f"Format: {format.upper()}\n"
        response += f"Method: {results_key}\n"
        response += f"Size: {file_size:,} bytes\n\n"

        response += "Next Steps:\n\n"
        if format == "csv":
            response += "- Open in Excel, R, or Stata for further analysis\n"
            response += (
                "- Import into statistical software using standard CSV readers\n"
            )
        elif format == "excel":
            response += "- Open in Excel for viewing and editing\n"
            response += "- Use multiple sheets for different result components\n"
        elif format == "latex":
            response += "- Include in your LaTeX document using \\input{}\n"
            response += "- Tables are publication-ready with standard formatting\n"
        elif format == "markdown":
            response += "- View in any markdown reader\n"
            response += "- Convert to HTML/PDF using pandoc\n"
        elif format == "json":
            response += "- Use for programmatic access\n"
            response += "- Archive for reproducibility\n"

        return response

    except Exception as e:
        logger.error(f"Error in export_results: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def export_comparison(
    methods: Optional[List[str]] = None,
    format: str = "markdown",
    filename: Optional[str] = None,
    include_diagnostics: bool = False,
) -> str:
    """
    Export comparison of multiple DID estimation methods.

    This tool exports a side-by-side comparison table of different DID methods,
    allowing you to compare ATT estimates, standard errors, and event study results.

    Args:
        methods: List of method keys to compare. Options:
            - None (default) - Compare all available methods
            - List of specific method keys

            Naming Convention:
            - Methods called directly: Use method name without prefix
              - "callaway_santanna" - Callaway & Sant'Anna (2021)
              - "sun_abraham" - Sun & Abraham (2021)
              - "bjs_imputation" - Borusyak, Jaravel & Spiess (2024)
              - "dcdh" - de Chaisemartin & D'Haultfoeuille (2020)
              - "gardner_two_stage" - Gardner (2022) Two-Stage
              - "gsynth" - Generalized Synthetic Control (Xu 2017)
              - "synthdid" - Synthetic DID (Arkhangelsky et al. 2019)
              - "efficient" - DISABLED (see KNOWN_ISSUES.md)

            - Methods called via workflow(): Use "workflow_" prefix
              - "workflow_callaway_santanna"
              - "workflow_sun_abraham"
              - "workflow_imputation_bjs"
              - etc.

            Tip: Set methods=None to automatically compare all available results.

        format: Export format ("markdown", "latex", "csv", "json")
        filename: Custom filename (optional, auto-generated if not provided)
        include_diagnostics: Include diagnostic comparison (default: False)

    Returns:
        Comparison report with all methods side-by-side

    Examples:
        >>> # Compare all available methods (auto-detects naming)
        >>> export_comparison()

        >>> # Compare specific direct method calls
        >>> export_comparison(
        ...     methods=["callaway_santanna", "sun_abraham", "bjs_imputation"],
        ...     format="latex",
        ...     filename="top3_methods.tex"
        ... )

        >>> # Compare workflow results
        >>> export_comparison(
        ...     methods=["workflow_callaway_santanna", "workflow_sun_abraham"],
        ...     format="csv"
        ... )

        >>> # Mix of direct and workflow results
        >>> export_comparison(
        ...     methods=["callaway_santanna", "workflow_sun_abraham"]
        ... )
    """
    try:
        analyzer = get_analyzer()

        # Get list of methods to compare
        if methods is None:
            # Get all available methods (excluding 'latest' and internal keys)
            methods = [
                k
                for k in analyzer.results.keys()
                if k not in ["latest", "diagnostics", "workflow"]
            ]

        if not methods:
            return "Error: No estimation results available. Please run at least one DID estimation method first."

        # Collect results for each method
        comparison_data = {}
        for method_key in methods:
            if method_key in analyzer.results:
                comparison_data[method_key] = analyzer.results[method_key]
            else:
                logger.warning(f"Method '{method_key}' not found in results, skipping")

        if not comparison_data:
            available_keys = [
                k
                for k in analyzer.results.keys()
                if k not in ["latest", "diagnostics", "workflow"]
            ]
            return f"""Error: None of the specified methods found.

Available methods:
{chr(10).join([f"- {k}" for k in available_keys])}

Tip: Use methods=None to export all available methods.
"""

        # Format comparison table based on format
        if format == "markdown":
            content = _format_comparison_as_markdown(
                comparison_data, include_diagnostics
            )
            extension = "md"
        elif format == "latex":
            content = _format_comparison_as_latex(comparison_data, include_diagnostics)
            extension = "tex"
        elif format == "csv":
            content = _format_comparison_as_csv(comparison_data, include_diagnostics)
            extension = "csv"
        elif format == "json":
            content = json.dumps(comparison_data, indent=2, default=str)
            extension = "json"
        else:
            return f"Error: Unsupported format '{format}'. Use: markdown, latex, csv, or json"

        # Prepare filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"did_comparison_{len(comparison_data)}methods_{timestamp}.{extension}"
            )
        elif not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"

        # Save file
        storage = get_storage()
        file_path = storage.base_dir / "exports" / filename

        # Ensure directory exists and resolve to absolute path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = file_path.resolve()  # Convert to absolute path

        logger.info(f"Exporting comparison to: {file_path}")
        logger.info(f"Storage base directory: {storage.base_dir.resolve()}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Verify file was written successfully
        if not file_path.exists():
            raise FileNotFoundError(
                f"File write reported success but file not found: {file_path}"
            )

        file_size = file_path.stat().st_size
        logger.info(f"File written successfully: {file_path} ({file_size:,} bytes)")

        # Generate success message
        response = f"Method Comparison Exported Successfully\n\n"
        response += f"File: {file_path}\n"
        response += f"Format: {format.upper()}\n"
        response += f"Methods Compared: {len(comparison_data)}\n"
        response += f"Method List: {', '.join(comparison_data.keys())}\n"
        response += f"Size: {file_size:,} bytes\n\n"

        response += "Comparison Includes:\n\n"
        response += "- Overall ATT estimates with standard errors and confidence intervals\n"
        response += "- Event study estimates for all periods\n"
        response += "- Statistical significance indicators\n"
        if include_diagnostics:
            response += "- Diagnostic information (if available)\n"

        response += "\nNext Steps:\n\n"
        if format == "markdown":
            response += "- View comparison table in any markdown reader\n"
            response += "- Convert to presentation slides or HTML\n"
        elif format == "latex":
            response += "- Include in your paper with \\input{}\n"
            response += "- Publication-ready comparison table\n"
        elif format == "csv":
            response += "- Analyze in Excel, R, or Stata\n"
            response += "- Easy to create custom visualizations\n"
        elif format == "json":
            response += "- Use for programmatic analysis\n"
            response += "- Archive for reproducibility\n"

        return response

    except Exception as e:
        logger.error(f"Error in export_comparison: {e}")
        return f"Error: {str(e)}"


# =============================================================================
# SENSITIVITY ANALYSIS TOOLS
# =============================================================================


@mcp.tool()
async def sensitivity_analysis(
    data_id: str = "current",
    method: str = "relative_magnitude",
    m_values: Optional[Union[List[float], str]] = None,
    event_time: int = 0,
    confidence_level: float = 0.95,
    estimator_method: str = "callaway_santanna",
) -> str:
    """
    Conduct HonestDiD sensitivity analysis for parallel trends assumption.

    Based on Rambachan & Roth (2023) "A More Credible Approach to Parallel Trends"

    Args:
        data_id: Dataset identifier (default: "current")
        method: "relative_magnitude" or "smoothness" constraints
        m_values: List of M values for sensitivity (default: [0.5, 1.0, 1.5, 2.0])
                 Can be a JSON array or string representation: "[0.5, 1.0, 1.5, 2.0]"
        event_time: Target event time for analysis (default: 0)
        confidence_level: Confidence level for intervals (default: 0.95)
        estimator_method: Base DID estimator ("callaway_santanna" or "sun_abraham")

    Returns:
        Formatted sensitivity analysis results with robustness assessment
    """
    # Validate and parse parameters using Pydantic model
    try:
        params = SensitivityAnalysisParams(
            data_id=data_id,
            method=method,
            m_values=m_values,
            event_time=event_time,
            confidence_level=confidence_level,
            estimator_method=estimator_method,
        )
        # Extract validated values
        m_values = params.m_values
        method = params.method
        confidence_level = params.confidence_level
    except Exception as e:
        return f"""Parameter validation error:

{str(e)}

Note: For m_values parameter, you can use:
- JSON array: [0.5, 1.0, 1.5, 2.0]
- String representation: "[0.5, 1.0, 1.5, 2.0]"
- Or omit it to use default values: [0.5, 1.0, 1.5, 2.0]

Example:

sensitivity_analysis(
    method="relative_magnitude",
    m_values=[0.5, 1.0, 1.5, 2.0],  # Or "[0.5, 1.0, 1.5, 2.0]"
    estimator_method="callaway_santanna"
)

"""

    try:
        if err := _require_data(): return err
        if err := _require_results(): return err
        if err := _require_r(): return err
        analyzer = get_analyzer()

        # Get latest results (primary estimator)
        latest_results = analyzer.results.get("latest")
        if not latest_results:
            return "Error: No valid estimation results found."

        # Extract event study data
        event_study = latest_results.get("event_study", {})
        if not event_study:
            return "Error: No event study results available for sensitivity analysis."

        # Prepare data for HonestDiD
        sorted_times = sorted(event_study.keys())
        pre_periods = [
            t for t in sorted_times if t < 0 and t != -1
        ]  # Exclude reference period
        post_periods = [t for t in sorted_times if t >= 0]

        # Check if we have sufficient pre-periods for sensitivity analysis
        if len(pre_periods) == 0:
            # Check if this is Efficient Estimator
            method_name = latest_results.get("method", "Unknown")
            if "Efficient" in method_name:
                return """Cannot perform HonestDiD sensitivity analysis with Efficient Estimator:

Issue: The staggered R package's Efficient Estimator only returns overall ATT by default, without event study estimates for individual time periods.

Why: The staggered package has a bug in its eventstudy mode that prevents extraction of multiple time periods (verified in testing).

Solution: Use an alternative estimator that provides full event study results:

1. Callaway & Sant'Anna (2021) RECOMMENDED
   
   estimate_callaway_santanna(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
   sensitivity_analysis(estimator_method="callaway_santanna")
   

2. Sun & Abraham (2021) ALTERNATIVE
   
   estimate_sun_abraham(formula="lemp ~ sunab(first.treat, year) | countyreal + year")
   sensitivity_analysis(estimator_method="sun_abraham")
   

Both methods provide full event study estimates and are fully compatible with HonestDiD sensitivity analysis.

Note: This is a known limitation of the staggered R package (v1.2.2), not our implementation.
"""
            else:
                return """Cannot perform HonestDiD sensitivity analysis:

HonestDiD requires pre-treatment periods (excluding the reference period -1) to test the parallel trends assumption.

Current data:
- Pre-treatment periods (excluding -1): None found
- Post-treatment periods: {0}

Recommendation:
- Ensure your data includes at least 2-3 pre-treatment periods before the treatment
- Run an estimator that provides event study estimates for periods before -1
- Try Callaway & Sant'Anna or Sun & Abraham methods which always provide full event studies
""".format(len(post_periods))

        elif len(pre_periods) == 1:
            logger.warning(
                "Only 1 pre-treatment period available for HonestDiD. Results may be less reliable."
            )

        # Extract coefficients (excluding reference period)
        betahat = []
        used_periods = []

        for t in pre_periods:
            betahat.append(event_study[t]["estimate"])
            used_periods.append(t)
        for t in post_periods:
            betahat.append(event_study[t]["estimate"])
            used_periods.append(t)

        betahat = np.array(betahat)

        # Extract variance-covariance matrix
        if "event_study_vcov" in latest_results:
            vcov_info = latest_results["event_study_vcov"]
            full_sigma = np.array(vcov_info["matrix"])
            vcov_periods = vcov_info["periods"]
            period_to_idx = {p: i for i, p in enumerate(vcov_periods)}
            used_indices = [
                period_to_idx[t] for t in used_periods if t in period_to_idx
            ]

            if len(used_indices) == len(used_periods):
                sigma = full_sigma[np.ix_(used_indices, used_indices)]
            else:
                sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])
        else:
            sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])

        # Default M values if not provided
        if m_values is None:
            m_values = [0.5, 1.0, 1.5, 2.0]

        # Run HonestDiD sensitivity analysis
        sensitivity_result = analyzer.r_estimators.honest_did_sensitivity_analysis(
            betahat=betahat,
            sigma=sigma,
            num_pre_periods=len(pre_periods),
            num_post_periods=len(post_periods),
            method=method,
            m_values=m_values,
            confidence_level=confidence_level,
        )

        if err := _check_result(sensitivity_result): return err

        # Format results
        method_desc = (
            "Smoothness Restrictions"
            if method == "smoothness"
            else "Relative Magnitude"
        )

        result_text = f"""
# HonestDiD Sensitivity Analysis Results

Method: {method_desc}
Confidence Level: {confidence_level * 100:.0f}%
Base Estimator: {estimator_method}
 Robust Confidence Intervals for Period {event_time}

"""

        # Show original confidence interval first
        if "original_ci" in sensitivity_result and sensitivity_result["original_ci"]:
            orig_ci = sensitivity_result["original_ci"]
            if (
                orig_ci.get("ci_lower") is not None
                and orig_ci.get("ci_upper") is not None
            ):
                result_text += f"""
 Original Estimate (No Violations Assumed)
- 95% CI: [{orig_ci["ci_lower"]:.4f}, {orig_ci["ci_upper"]:.4f}]
- Statistically significant: {"Yes" if orig_ci["ci_lower"] > 0 or orig_ci["ci_upper"] < 0 else "No"}

"""

        # Add results for each M value from robust_intervals
        if "robust_intervals" in sensitivity_result:
            robust_intervals = sensitivity_result["robust_intervals"]
            for m_val in m_values:
                m_key = f"M_{m_val}"  # Note: Using "M_" not "M="
                if m_key in robust_intervals:
                    m_result = robust_intervals[m_key]
                    ci_lower = m_result["ci_lower"]
                    ci_upper = m_result["ci_upper"]
                    contains_zero = m_result["contains_zero"]
                    result_text += f"""
 M = {m_val} ({method_desc})
- Robust 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
- Statistically significant: {"No (confidence interval includes zero)" if contains_zero else "Yes"}
- Width: {ci_upper - ci_lower:.4f}
"""

        # Add breakdown point if available
        if (
            "breakdown_point" in sensitivity_result
            and sensitivity_result["breakdown_point"] is not None
        ):
            breakdown = sensitivity_result["breakdown_point"]
            result_text += f"""
 Breakdown Point Analysis

M̄ (Breakdown Value): {breakdown:.4f}

This is the minimum violation magnitude where the confidence interval first includes zero.

Interpretation:
- M̄ = {breakdown:.4f} means your results remain significant as long as parallel trends violations are smaller than {breakdown:.1%} of the reference trend
- Larger M̄ = More Robust: Results can tolerate bigger violations before losing significance
- Smaller M̄ = Less Robust: Results are sensitive to even small violations

"""
            if breakdown >= 2.0:
                result_text += "Very Robust: Results remain significant even with large violations\n"
            elif breakdown >= 1.0:
                result_text += "Moderately Robust: Results significant for moderate violations\n"
            elif breakdown >= 0.5:
                result_text += "Somewhat Sensitive: Results vulnerable to moderate violations\n"
            else:
                result_text += "Highly Sensitive: Results vulnerable to even small violations\n"
        else:
            result_text += """
 Breakdown Point Analysis

M̄ (Breakdown Value): Not reached within tested range

Very Robust: The confidence interval remains significant for all tested violation magnitudes.
This suggests strong robustness to parallel trends violations.
"""

        return result_text

    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def power_analysis(
    data_id: str = "current",
    target_power: float = 0.8,
    alpha: float = 0.05,
    trend_type: str = "linear",
) -> str:
    """
    Conduct power analysis for pre-trends tests using pretrends package.

    Based on Roth (2022) "Event-Study Estimates after Testing for Parallel Trends"

    Args:
        data_id: Dataset identifier (default: "current")
        target_power: Target statistical power (default: 0.8)
        alpha: Significance level (default: 0.05)
        trend_type: Type of trend violation ("linear" or "non_linear")

    Returns:
        Power analysis results with minimum detectable effect sizes
    """
    try:
        if err := _require_data(): return err
        if err := _require_results(): return err
        if err := _require_r(): return err
        analyzer = get_analyzer()

        # Get latest estimation results (primary estimator)
        latest_results = analyzer.results["latest"]

        # Check if we have event study results
        if "event_study" not in latest_results:
            return "Error: No event study results available. Please run an event study estimator (CS, SA, etc.) first."

        # Prepare matrices (delegates to single source of truth)
        try:
            matrices = analyzer._prepare_event_study_matrices(latest_results)
        except ValueError as e:
            return f"Error: {e}"

        betahat = matrices["betahat"]
        sigma = matrices["sigma"]
        time_vec = matrices["time_vec"]
        logger.info(f"Using covariance method: {matrices['covariance_method']}")

        # Run pretrends power analysis via R estimators
        r_estimators = analyzer.r_estimators

        # Calculate minimal detectable slope for target power
        power_result = r_estimators.pretrends_power_analysis(
            betahat=betahat,
            sigma=sigma,
            time_vec=time_vec,
            reference_period=-1,
            target_power=target_power,
            alpha=alpha,
            analyze_slope=None,  # Just ex ante analysis for now
        )

        if err := _check_result(power_result): return err

        # Format results for user display
        return _format_power_analysis_results(power_result, trend_type)

    except Exception as e:
        logger.error(f"Error in power analysis: {e}")
        return f"Error: {str(e)}"


def _format_power_analysis_results(
    power_result: Dict[str, Any], trend_type: str
) -> str:
    """Format pretrends power analysis results for user display."""

    if err := _check_result(power_result): return err

    minimal_slope = power_result["minimal_detectable_slope"]
    target_power = power_result["target_power"]
    interpretation = power_result["interpretation"]

    # Create formatted output
    output_lines = [
        "Pretrends Power Analysis Results",
        "",
        f"Analysis Method: {trend_type.replace('_', ' ').title()} Trend Violations",
        f"Target Power: {target_power * 100}%",
        f"Time Periods Analyzed: {power_result['num_periods']}",
        f"Reference Period: {power_result['reference_period']}",
        "",
        "Minimal Detectable Effect",
    ]

    output_lines.extend(
        [
            f"- Slope: {minimal_slope:.6f}",
            f"- Interpretation: {interpretation['minimal_slope']}",
            "",
            "Power Assessment",
            f"- Assessment: {interpretation['power_assessment']}",
            f"- Recommendation: {interpretation['recommendation']}",
            "",
        ]
    )

    # Add ex post analysis if available
    if power_result.get("power_analysis") and "ex_post" in interpretation:
        output_lines.extend(
            ["Specific Slope Analysis", f"- {interpretation['ex_post']}", ""]
        )

    # Add practical interpretation
    output_lines.extend(
        [
            "What This Means",
            "",
            "Ex Ante Power: This analysis tells you the magnitude of linear pre-trend violation",
            f"that your test would detect {target_power * 100}% of the time. If violations smaller than",
            f"{minimal_slope:.6f} exist, your pre-trends test will often miss them.",
            "",
            "Policy Implications:",
        ]
    )

    if minimal_slope <= 0.01:
        output_lines.extend(
            [
                "Your test has good power to detect small violations",
                "Negative pre-trends test provides reasonable evidence for parallel trends",
            ]
        )
    elif minimal_slope <= 0.05:
        output_lines.extend(
            [
                "Your test has moderate power - may miss small violations",
                "Consider additional robustness checks (e.g., HonestDiD sensitivity analysis)",
            ]
        )
    else:
        output_lines.extend(
            [
                "Your test has low power - likely to miss meaningful violations",
                "Strongly recommend sensitivity analysis before drawing causal conclusions",
            ]
        )

    output_lines.extend(
        [
            "",
            "Next Steps",
            "",
            "1. If power is adequate: Proceed with causal interpretation",
            "2. If power is low: Use sensitivity_analysis to assess robustness",
            "3. For comprehensive analysis: Combine with HonestDiD sensitivity bounds",
            "",
            'Reference: Roth (2022) "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends"',
        ]
    )

    return "\n".join(output_lines)


# =============================================================================
# EXPORT HELPER FUNCTIONS
# =============================================================================


def _format_results_as_csv(data: Dict) -> str:
    """Format results as CSV."""
    import pandas as pd
    from io import StringIO

    output = StringIO()
    results = data.get("results", {})

    # Event study results
    if "event_study" in results:
        df = pd.DataFrame(results["event_study"]).T
        df.index.name = "period"
        df.to_csv(output)

    # Overall ATT
    if "overall_att" in results:
        output.write("\n\nOverall ATT\n")
        att = results["overall_att"]
        output.write(f"estimate,{att['estimate']}\n")
        output.write(f"std_error,{att.get('se', att.get('std_error', 'N/A'))}\n")
        output.write(f"ci_lower,{att['ci_lower']}\n")
        output.write(f"ci_upper,{att['ci_upper']}\n")
        output.write(f"p_value,{att.get('pvalue', att.get('p_value', 'N/A'))}\n")

    return output.getvalue()


def _format_results_as_latex(data: Dict) -> str:
    """Format results as LaTeX table."""
    results = data.get("results", {})
    output = []

    # Header
    output.append("% DID Analysis Results")
    output.append("% Generated by ChatDiD\n")

    # Event study table
    if "event_study" in results:
        output.append("\\begin{table}[htbp]")
        output.append("\\centering")
        output.append("\\caption{Event Study Estimates}")
        output.append("\\begin{tabular}{lcccc}")
        output.append("\\hline\\hline")
        output.append("Period & Estimate & Std. Error & CI Lower & CI Upper \\\\")
        output.append("\\hline")

        for period, est in sorted(results["event_study"].items()):
            output.append(
                f"{period} & {est['estimate']:.4f} & {est['se']:.4f} & "
                f"{est['ci_lower']:.4f} & {est['ci_upper']:.4f} \\\\"
            )

        output.append("\\hline\\hline")
        output.append("\\end{tabular}")
        output.append("\\end{table}\n")

    # Overall ATT
    if "overall_att" in results:
        att = results["overall_att"]
        output.append("\\begin{table}[htbp]")
        output.append("\\centering")
        output.append("\\caption{Overall Average Treatment Effect}")
        output.append("\\begin{tabular}{lc}")
        output.append("\\hline\\hline")
        se_val = att.get("se", att.get("std_error", None))
        p_val = att.get("pvalue", att.get("p_value", None))
        output.append(f"ATT & {att['estimate']:.4f}")
        output.append(f"    & ({se_val:.4f}) \\\\" if se_val else "    & (N/A) \\\\")
        output.append(f"95\\% CI & [{att['ci_lower']:.4f}, {att['ci_upper']:.4f}] \\\\")
        output.append(f"p-value & {p_val:.4f} \\\\" if p_val else "p-value & N/A \\\\")
        output.append("\\hline\\hline")
        output.append("\\end{tabular}")
        output.append("\\end{table}")

    return "\n".join(output)


def _format_results_as_markdown(data: Dict) -> str:
    """Format results as Markdown."""
    results = data.get("results", {})
    output = []

    output.append("# DID Analysis Results")
    output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Method information
    if "method" in results:
        output.append(f"Method: {results['method']}\n")

    # Overall ATT
    if "overall_att" in results:
        att = results["overall_att"]
        output.append("Overall Average Treatment Effect\n")
        output.append(f"- Estimate: {att['estimate']:.4f}")
        se_val = att.get("se", att.get("std_error", None))
        p_val = att.get("pvalue", att.get("p_value", None))
        output.append(
            f"- Std. Error: {se_val:.4f}" if se_val else "- Std. Error: N/A"
        )
        output.append(f"- 95% CI: [{att['ci_lower']:.4f}, {att['ci_upper']:.4f}]")
        output.append(
            f"- p-value: {p_val:.4f}\n" if p_val else "- p-value: N/A\n"
        )

    # Event study
    if "event_study" in results:
        output.append("Event Study Estimates\n")
        output.append("| Period | Estimate | Std. Error | CI Lower | CI Upper |")
        output.append("|--------|----------|------------|----------|----------|")

        for period, est in sorted(results["event_study"].items()):
            output.append(
                f"| {period} | {est['estimate']:.4f} | {est['se']:.4f} | "
                f"{est['ci_lower']:.4f} | {est['ci_upper']:.4f} |"
            )
        output.append("")

    # Diagnostics if included
    if "diagnostics" in data:
        output.append("Diagnostics\n")
        diag = data["diagnostics"]
        if isinstance(diag, dict):
            for key, value in diag.items():
                output.append(f"- {key}: {value}")

    return "\n".join(output)


def _format_results_as_excel(data: Dict) -> bytes:
    """Format results for Excel export (returns placeholder)."""
    # This is handled directly in export_results function
    # Return empty bytes as placeholder
    return b""


# =============================================================================
# COMPARISON EXPORT HELPER FUNCTIONS
# =============================================================================


def _format_comparison_as_markdown(
    comparison_data: Dict[str, Dict], include_diagnostics: bool = False
) -> str:
    """Format method comparison as Markdown table."""
    output = []
    output.append("# DID Methods Comparison Report")
    output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"\nNumber of Methods: {len(comparison_data)}")
    output.append("")

    # Overall ATT Comparison
    output.append("Overall Average Treatment Effects (ATT)")
    output.append("")
    output.append("| Method | Estimate | Std. Error | 95% CI | p-value | Significant |")
    output.append("|--------|----------|------------|--------|---------|-------------|")

    for method_name, results in comparison_data.items():
        if "overall_att" in results:
            att = results["overall_att"]
            est = att.get("estimate", "N/A")
            se = att.get("se", att.get("std_error", "N/A"))
            ci_lower = att.get("ci_lower", "N/A")
            ci_upper = att.get("ci_upper", "N/A")
            pval = att.get("pvalue", att.get("p_value", "N/A"))

            # Format values
            if isinstance(est, (int, float)):
                est_str = f"{est:.4f}"
            else:
                est_str = str(est)

            if isinstance(se, (int, float)):
                se_str = f"{se:.4f}"
            else:
                se_str = str(se)

            if isinstance(ci_lower, (int, float)) and isinstance(
                ci_upper, (int, float)
            ):
                ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            else:
                ci_str = "N/A"

            if isinstance(pval, (int, float)):
                pval_str = f"{pval:.4f}"
                sig_str = "Yes" if pval < 0.05 else "No"
            else:
                pval_str = str(pval)
                sig_str = "N/A"

            # Friendly method name
            method_display = method_name.replace("_", " ").title()

            output.append(
                f"| {method_display} | {est_str} | {se_str} | {ci_str} | {pval_str} | {sig_str} |"
            )

    output.append("")

    # Event Study Comparison (if available)
    # Find common event times across all methods
    all_event_times = set()
    for results in comparison_data.values():
        if "event_study" in results and isinstance(results["event_study"], dict):
            all_event_times.update(results["event_study"].keys())

    if all_event_times:
        sorted_times = sorted(
            all_event_times,
            key=lambda x: float(x)
            if isinstance(x, (int, float, str))
            and str(x).replace("-", "").replace(".", "").isdigit()
            else 999,
        )

        output.append("Event Study Estimates by Period")
        output.append("")

        # Create header
        header = "| Period |"
        separator = "|--------|"
        for method_name in comparison_data.keys():
            method_display = method_name.replace("_", " ").title()
            header += f" {method_display} |"
            separator += "----------|"
        output.append(header)
        output.append(separator)

        # Create rows for each event time
        for event_time in sorted_times:
            row = f"| {event_time} |"
            for method_name, results in comparison_data.items():
                if "event_study" in results and event_time in results["event_study"]:
                    est_data = results["event_study"][event_time]
                    if isinstance(est_data, dict):
                        est = est_data.get("estimate", "N/A")
                        if isinstance(est, (int, float)):
                            cell = f" {est:.4f} |"
                        else:
                            cell = f" {est} |"
                    else:
                        cell = " N/A |"
                else:
                    cell = " - |"
                row += cell
            output.append(row)

    output.append("")
    output.append("---")
    output.append("")
    output.append("Legend:")
    output.append("- Statistically significant at 5% level")
    output.append("- Not statistically significant")
    output.append("- N/A: Not available for this method")

    return "\n".join(output)


def _format_comparison_as_latex(
    comparison_data: Dict[str, Dict], include_diagnostics: bool = False
) -> str:
    """Format method comparison as LaTeX table."""
    output = []
    output.append("% DID Methods Comparison")
    output.append("% Generated by ChatDiD")
    output.append("")

    # Overall ATT comparison table
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{Comparison of DID Estimation Methods - Overall ATT}")
    output.append("\\begin{tabular}{lccccl}")
    output.append("\\hline\\hline")
    output.append(
        "Method & Estimate & Std. Error & 95\\% CI & p-value & Significant \\\\"
    )
    output.append("\\hline")

    for method_name, results in comparison_data.items():
        if "overall_att" in results:
            att = results["overall_att"]
            est = att.get("estimate", "N/A")
            se = att.get("se", att.get("std_error", "N/A"))
            ci_lower = att.get("ci_lower", "N/A")
            ci_upper = att.get("ci_upper", "N/A")
            pval = att.get("pvalue", att.get("p_value", "N/A"))

            # Format values
            if isinstance(est, (int, float)):
                est_str = f"{est:.4f}"
            else:
                est_str = str(est)

            if isinstance(se, (int, float)):
                se_str = f"({se:.4f})"
            else:
                se_str = str(se)

            if isinstance(ci_lower, (int, float)) and isinstance(
                ci_upper, (int, float)
            ):
                ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            else:
                ci_str = "N/A"

            if isinstance(pval, (int, float)):
                pval_str = f"{pval:.4f}"
                sig_str = "Yes" if pval < 0.05 else "No"
            else:
                pval_str = str(pval)
                sig_str = "N/A"

            # Format method name for LaTeX
            method_display = (
                method_name.replace("_", " ")
                .title()
                .replace("Bjs", "BJS")
                .replace("Dcdh", "DCDH")
            )

            output.append(
                f"{method_display} & {est_str} & {se_str} & {ci_str} & {pval_str} & {sig_str} \\\\"
            )

    output.append("\\hline\\hline")
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    output.append("")

    # Event study comparison (simplified - show only key periods)
    # Find common event times
    all_event_times = set()
    for results in comparison_data.values():
        if "event_study" in results and isinstance(results["event_study"], dict):
            all_event_times.update(results["event_study"].keys())

    if all_event_times:
        sorted_times = sorted(
            all_event_times,
            key=lambda x: float(x)
            if isinstance(x, (int, float, str))
            and str(x).replace("-", "").replace(".", "").isdigit()
            else 999,
        )

        # Limit to max 10 periods for readability
        if len(sorted_times) > 10:
            # Select key periods: first 3, middle, last 3, and period 0
            key_times = sorted_times[:3] + [0] + sorted_times[-3:]
            key_times = sorted(set(key_times))
        else:
            key_times = sorted_times

        num_methods = len(comparison_data)
        col_spec = "l" + "c" * num_methods

        output.append("\\begin{table}[htbp]")
        output.append("\\centering")
        output.append("\\caption{Event Study Estimates - Selected Periods}")
        output.append(f"\\begin{{tabular}}{{{col_spec}}}")
        output.append("\\hline\\hline")

        # Header
        header = "Period"
        for method_name in comparison_data.keys():
            method_display = (
                method_name.replace("_", " ")
                .title()
                .replace("Bjs", "BJS")
                .replace("Dcdh", "DCDH")
            )
            header += f" & {method_display}"
        header += " \\\\"
        output.append(header)
        output.append("\\hline")

        # Rows
        for event_time in key_times:
            row = f"{event_time}"
            for method_name, results in comparison_data.items():
                if "event_study" in results and event_time in results["event_study"]:
                    est_data = results["event_study"][event_time]
                    if isinstance(est_data, dict):
                        est = est_data.get("estimate", "N/A")
                        if isinstance(est, (int, float)):
                            row += f" & {est:.4f}"
                        else:
                            row += f" & {est}"
                    else:
                        row += " & N/A"
                else:
                    row += " & --"
            row += " \\\\"
            output.append(row)

        output.append("\\hline\\hline")
        output.append("\\end{tabular}")
        output.append("\\end{table}")

    return "\n".join(output)


def _format_comparison_as_csv(
    comparison_data: Dict[str, Dict], include_diagnostics: bool = False
) -> str:
    """Format method comparison as CSV."""
    import pandas as pd
    from io import StringIO

    output = StringIO()

    # Overall ATT comparison
    att_data = []
    for method_name, results in comparison_data.items():
        if "overall_att" in results:
            att = results["overall_att"]
            row = {
                "method": method_name,
                "estimate": att.get("estimate", ""),
                "std_error": att.get("se", att.get("std_error", "")),
                "ci_lower": att.get("ci_lower", ""),
                "ci_upper": att.get("ci_upper", ""),
                "p_value": att.get("pvalue", att.get("p_value", "")),
            }
            att_data.append(row)

    if att_data:
        df_att = pd.DataFrame(att_data)
        output.write("# Overall ATT Comparison\n")
        df_att.to_csv(output, index=False)
        output.write("\n")

    # Event study comparison
    all_event_times = set()
    for results in comparison_data.values():
        if "event_study" in results and isinstance(results["event_study"], dict):
            all_event_times.update(results["event_study"].keys())

    if all_event_times:
        sorted_times = sorted(
            all_event_times,
            key=lambda x: float(x)
            if isinstance(x, (int, float, str))
            and str(x).replace("-", "").replace(".", "").isdigit()
            else 999,
        )

        event_study_data = []
        for event_time in sorted_times:
            row = {"period": event_time}
            for method_name, results in comparison_data.items():
                if "event_study" in results and event_time in results["event_study"]:
                    est_data = results["event_study"][event_time]
                    if isinstance(est_data, dict):
                        row[f"{method_name}_estimate"] = est_data.get("estimate", "")
                        row[f"{method_name}_se"] = est_data.get(
                            "se", est_data.get("std_error", "")
                        )
                    else:
                        row[f"{method_name}_estimate"] = ""
                        row[f"{method_name}_se"] = ""
                else:
                    row[f"{method_name}_estimate"] = ""
                    row[f"{method_name}_se"] = ""
            event_study_data.append(row)

        if event_study_data:
            df_event = pd.DataFrame(event_study_data)
            output.write("# Event Study Comparison\n")
            df_event.to_csv(output, index=False)

    return output.getvalue()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

# When using FastMCP CLI (fastmcp run), no explicit main() is needed
# FastMCP automatically handles the server lifecycle and asyncio event loop
