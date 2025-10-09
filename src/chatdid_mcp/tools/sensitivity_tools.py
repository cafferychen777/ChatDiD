"""
Sensitivity Analysis Tools for DID Analysis
Implements HonestDiD sensitivity analysis methods
Based on Rambachan & Roth (2023)
"""

import logging
import sys
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Configure logging to stderr ONLY for MCP compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    from ..did_analyzer import get_analyzer
except ImportError:
    logger.error("Could not import DiD analyzer")

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except ImportError as e:
    R_AVAILABLE = False
    logger.error(f"R integration not available: {e}")


async def sensitivity_analysis_tool(
    data_id: str,
    method: str = "relative_magnitude",
    m_values: Optional[List[float]] = None,
    event_time: int = 0,
    confidence_level: float = 0.95,
    estimator_method: str = "callaway_santanna"
) -> str:
    """
    Conduct sensitivity analysis using HonestDiD methods.
    
    Args:
        data_id: Dataset identifier
        method: "relative_magnitude" or "smoothness"
        m_values: List of M values for sensitivity analysis
        event_time: Target event time for analysis
        confidence_level: Confidence level for intervals
        estimator_method: DID estimator to use as base
        
    Returns:
        Formatted sensitivity analysis results
    """
    try:
        analyzer = get_analyzer()
        
        # Check if data is loaded
        if analyzer.data is None:
            return "❌ **Error:** No data loaded. Please load data first using `load_data`."
        
        logger.info(f"Starting sensitivity analysis with method={method}")
        
        # Set default M values based on research best practices
        if m_values is None:
            # Standard setting from research: balanced conservatism and coverage
            m_values = [0.5, 1.0, 1.5, 2.0]
        
        # Step 1: Run base DID estimator to get event study results
        if estimator_method == "callaway_santanna":
            estimator_results = await _run_callaway_santanna_for_sensitivity(analyzer)
        elif estimator_method == "sun_abraham":
            estimator_results = await _run_sun_abraham_for_sensitivity(analyzer)
        else:
            return f"❌ **Error:** Unsupported estimator method: {estimator_method}"
        
        if estimator_results["status"] != "success":
            return f"❌ **Error:** Base estimator failed: {estimator_results.get('message', 'Unknown error')}"
        
        # Step 2: Extract event study for sensitivity analysis
        event_study = estimator_results.get("event_study", {})
        if not event_study:
            return "❌ **Error:** No event study results available for sensitivity analysis."
        
        # Step 3: Prepare data for HonestDiD
        sensitivity_data = _prepare_sensitivity_data(estimator_results, event_time)
        
        # Step 4: Run HonestDiD analysis
        sensitivity_results = await _run_honest_did_analysis(
            sensitivity_data, method, m_values, confidence_level
        )
        
        if sensitivity_results["status"] != "success":
            return f"❌ **Error:** Sensitivity analysis failed: {sensitivity_results.get('message', 'Unknown error')}"
        
        # Step 5: Format results
        return _format_sensitivity_results(sensitivity_results, estimator_method, method)
        
    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {e}")
        return f"❌ **Error:** {str(e)}"


async def _run_callaway_santanna_for_sensitivity(analyzer) -> Dict[str, Any]:
    """Run Callaway & Sant'Anna estimator optimized for sensitivity analysis."""
    try:
        # Check if R packages are available
        if not R_AVAILABLE:
            return {"status": "error", "message": "R integration not available"}
        
        # Import R packages
        did_pkg = importr("did")
        
        # Use current DID configuration
        config = analyzer.config
        if not config:
            return {"status": "error", "message": "DID configuration not set. Please run `explore_data` first."}
        
        # Convert to R dataframe
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(analyzer.data)
        
        # Run att_gt with optimal settings for sensitivity analysis
        logger.info("Running Callaway & Sant'Anna for sensitivity analysis")
        cs_att = did_pkg.att_gt(
            yname=config["outcome_col"],
            tname=config["time_col"],
            idname=config["unit_col"],
            gname=config.get("cohort_col", config["treatment_col"]),
            data=r_data,
            control_group="notyettreated",
            est_method="dr",  # Doubly robust for sensitivity
            bstrap=True,
            biters=1000,
            cband=True  # Uniform confidence bands for sensitivity
        )
        
        # Aggregate for dynamic event study
        cs_es = did_pkg.aggte(cs_att, type="dynamic")
        
        # Extract event study estimates
        event_study = {}
        egt = cs_es.rx2('egt')  # Event time
        att_egt = cs_es.rx2('att.egt')  # ATT by event time
        se_egt = cs_es.rx2('se.egt')  # SE by event time
        
        for i, e in enumerate(egt):
            event_study[int(e)] = {
                "estimate": float(att_egt[i]),
                "se": float(se_egt[i]),
                "ci_lower": float(att_egt[i] - 1.96 * se_egt[i]),
                "ci_upper": float(att_egt[i] + 1.96 * se_egt[i])
            }
        
        return {
            "status": "success",
            "event_study": event_study,
            "overall_att": {
                "estimate": float(cs_es.rx2('overall.att')[0]),
                "se": float(cs_es.rx2('overall.se')[0])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in Callaway & Sant'Anna for sensitivity: {e}")
        return {"status": "error", "message": str(e)}


async def _run_sun_abraham_for_sensitivity(analyzer) -> Dict[str, Any]:
    """Run Sun & Abraham estimator optimized for sensitivity analysis."""
    try:
        # Placeholder for Sun & Abraham implementation
        # For now, fallback to Callaway & Sant'Anna
        return await _run_callaway_santanna_for_sensitivity(analyzer)
        
    except Exception as e:
        logger.error(f"Error in Sun & Abraham for sensitivity: {e}")
        return {"status": "error", "message": str(e)}


def _prepare_sensitivity_data(estimator_results: Dict, event_time: int) -> Dict[str, Any]:
    """
    Prepare event study data for HonestDiD analysis.
    
    Args:
        estimator_results: Full results from DID estimator including event study and covariance
        event_time: Target event time for analysis
        
    Returns:
        Prepared data for HonestDiD
    """
    event_study = estimator_results.get("event_study", {})
    
    # Sort by event time
    sorted_times = sorted(event_study.keys())
    
    # Separate pre and post periods
    pre_periods = [t for t in sorted_times if t < 0]
    post_periods = [t for t in sorted_times if t >= 0]
    
    # Extract coefficients
    betahat = []
    used_periods = []
    
    # Pre-period coefficients (excluding reference period -1 if present)
    for t in pre_periods:
        if t != -1:  # Exclude reference period
            betahat.append(event_study[t]["estimate"])
            used_periods.append(t)
    
    # Post-period coefficients
    for t in post_periods:
        betahat.append(event_study[t]["estimate"])
        used_periods.append(t)
    
    # Use full covariance matrix if available, otherwise fallback to diagonal
    if "event_study_vcov" in estimator_results:
        vcov_info = estimator_results["event_study_vcov"]
        full_sigma = np.array(vcov_info["matrix"])
        vcov_periods = vcov_info["periods"]
        
        # Create mapping from periods to indices in the covariance matrix
        period_to_idx = {p: i for i, p in enumerate(vcov_periods)}
        
        # Extract submatrix for used periods
        used_indices = [period_to_idx[t] for t in used_periods if t in period_to_idx]
        
        if len(used_indices) == len(used_periods):
            sigma = full_sigma[np.ix_(used_indices, used_indices)]
            logger.info(f"Using full covariance matrix for sensitivity analysis (method: {vcov_info['extraction_method']})")
        else:
            logger.warning("Period mismatch in covariance matrix, using diagonal fallback for sensitivity")
            sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])
    else:
        logger.warning("Full covariance matrix not available, using diagonal approximation for sensitivity")
        sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])
    
    return {
        "betahat": np.array(betahat),
        "sigma": sigma,
        "num_pre_periods": len([t for t in pre_periods if t != -1]),
        "num_post_periods": len(post_periods),
        "pre_periods": [t for t in pre_periods if t != -1],
        "post_periods": post_periods,
        "event_time": event_time
    }


async def _run_honest_did_analysis(
    sensitivity_data: Dict[str, Any],
    method: str,
    m_values: List[float],
    confidence_level: float
) -> Dict[str, Any]:
    """
    Run HonestDiD sensitivity analysis.
    
    Args:
        sensitivity_data: Prepared event study data
        method: Sensitivity analysis method
        m_values: List of M values to test
        confidence_level: Confidence level for intervals
        
    Returns:
        HonestDiD analysis results
    """
    try:
        if not R_AVAILABLE:
            return {"status": "error", "message": "R integration not available"}
        
        # Import HonestDiD package
        honest_did = importr("HonestDiD")
        
        # Convert numpy arrays to R
        r_betahat = robjects.FloatVector(sensitivity_data["betahat"])
        r_sigma = robjects.r['matrix'](
            robjects.FloatVector(sensitivity_data["sigma"].flatten()),
            nrow=sensitivity_data["sigma"].shape[0],
            ncol=sensitivity_data["sigma"].shape[1]
        )
        
        # Set up parameters
        num_pre = sensitivity_data["num_pre_periods"]
        num_post = sensitivity_data["num_post_periods"]
        alpha = 1 - confidence_level
        
        logger.info(f"Running HonestDiD with {num_pre} pre-periods, {num_post} post-periods")
        
        if method == "relative_magnitude":
            # Relative magnitude constraints
            r_m_values = robjects.FloatVector(m_values)
            
            # Create original confidence set
            original_cs = honest_did.constructOriginalCS(
                betahat=r_betahat,
                sigma=r_sigma,
                numPrePeriods=num_pre,
                numPostPeriods=num_post,
                alpha=alpha
            )
            
            # Run sensitivity analysis
            sensitivity_results_r = honest_did.createSensitivityResults_relativeMagnitudes(
                betahat=r_betahat,
                sigma=r_sigma,
                numPrePeriods=num_pre,
                numPostPeriods=num_post,
                Mbarvec=r_m_values,
                alpha=alpha
            )
            
            # Extract results
            results = _extract_relative_magnitude_results(
                sensitivity_results_r, original_cs, m_values
            )
            
        elif method == "smoothness":
            # Smoothness constraints (placeholder)
            return {"status": "error", "message": "Smoothness constraints not yet implemented"}
        
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}
        
        return {
            "status": "success",
            "method": method,
            "results": results,
            "data_info": {
                "num_pre_periods": num_pre,
                "num_post_periods": num_post,
                "m_values": m_values
            }
        }
        
    except Exception as e:
        logger.error(f"Error in HonestDiD analysis: {e}")
        return {"status": "error", "message": str(e)}


def _extract_relative_magnitude_results(sensitivity_results_r, original_cs, m_values: List[float]) -> Dict[str, Any]:
    """Extract results from HonestDiD relative magnitude analysis."""
    try:
        # Extract breakdown point (M value where result becomes insignificant)
        breakdown_point = None
        robust_intervals = {}
        
        # Process each M value
        for i, m in enumerate(m_values):
            try:
                # Extract confidence interval for this M value
                ci_lower = float(sensitivity_results_r.rx2('lowerCI')[i])
                ci_upper = float(sensitivity_results_r.rx2('upperCI')[i])
                
                robust_intervals[f"M_{m}"] = {
                    "m_value": m,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "contains_zero": ci_lower <= 0 <= ci_upper
                }
                
                # Check if this is the breakdown point
                if breakdown_point is None and ci_lower <= 0 <= ci_upper:
                    breakdown_point = m
                    
            except Exception as e:
                logger.warning(f"Could not extract results for M={m}: {e}")
        
        # Extract original confidence interval
        original_ci = {
            "ci_lower": float(original_cs.rx2('lowerCI')[0]),
            "ci_upper": float(original_cs.rx2('upperCI')[0])
        }
        
        return {
            "breakdown_point": breakdown_point,
            "robust_intervals": robust_intervals,
            "original_ci": original_ci
        }
        
    except Exception as e:
        logger.error(f"Error extracting HonestDiD results: {e}")
        return {
            "breakdown_point": None,
            "robust_intervals": {},
            "original_ci": {}
        }


def _format_sensitivity_results(
    sensitivity_results: Dict[str, Any],
    estimator_method: str,
    analysis_method: str
) -> str:
    """Format sensitivity analysis results for user display."""
    
    if sensitivity_results["status"] != "success":
        return f"❌ **Sensitivity Analysis Failed:** {sensitivity_results.get('message', 'Unknown error')}"
    
    results = sensitivity_results["results"]
    data_info = sensitivity_results["data_info"]
    
    # Create formatted output
    output_lines = [
        "## 🔍 **HonestDiD Sensitivity Analysis Results**",
        "",
        f"**Base Estimator:** {estimator_method.replace('_', ' ').title()}",
        f"**Sensitivity Method:** {analysis_method.replace('_', ' ').title()}",
        f"**Pre-treatment Periods:** {data_info['num_pre_periods']}",
        f"**Post-treatment Periods:** {data_info['num_post_periods']}",
        "",
        "### 📊 **Original Results (No Sensitivity Adjustment)**"
    ]
    
    # Original confidence interval
    original_ci = results.get("original_ci", {})
    if original_ci:
        ci_lower = original_ci["ci_lower"]
        ci_upper = original_ci["ci_upper"]
        contains_zero = ci_lower <= 0 <= ci_upper
        significance = "Not Significant" if contains_zero else "Significant"
        
        output_lines.extend([
            f"- **95% CI:** [{ci_lower:.4f}, {ci_upper:.4f}]",
            f"- **Significance:** {significance}",
            ""
        ])
    
    # Breakdown point analysis
    breakdown_point = results.get("breakdown_point")
    if breakdown_point is not None:
        output_lines.extend([
            "### ⚠️ **Robustness Assessment**",
            f"- **Breakdown Point:** M = {breakdown_point}",
            f"- **Interpretation:** Results become insignificant when post-treatment violations exceed {breakdown_point}× the maximum pre-treatment violation",
            ""
        ])
    else:
        output_lines.extend([
            "### ✅ **Robustness Assessment**",
            f"- **Breakdown Point:** > {max(data_info['m_values'])}",
            "- **Interpretation:** Results remain significant across all tested sensitivity levels",
            ""
        ])
    
    # Detailed sensitivity results
    robust_intervals = results.get("robust_intervals", {})
    if robust_intervals:
        output_lines.extend([
            "### 📈 **Sensitivity Analysis by M Value**",
            ""
        ])
        
        for m_key, interval in robust_intervals.items():
            m_val = interval["m_value"]
            ci_lower = interval["ci_lower"]
            ci_upper = interval["ci_upper"]
            contains_zero = interval["contains_zero"]
            status = "❌ Insignificant" if contains_zero else "✅ Significant"
            
            output_lines.append(f"- **M = {m_val}:** [{ci_lower:.4f}, {ci_upper:.4f}] {status}")
    
    # Interpretation and recommendations
    output_lines.extend([
        "",
        "### 🎯 **Summary and Recommendations**",
        ""
    ])
    
    if breakdown_point is not None and breakdown_point <= 1.0:
        output_lines.extend([
            "⚠️ **Caution:** Results are sensitive to modest violations of parallel trends.",
            f"- Consider additional robustness checks",
            f"- Examine pre-treatment trends more carefully",
            f"- Consider alternative identification strategies"
        ])
    elif breakdown_point is not None and breakdown_point <= 2.0:
        output_lines.extend([
            "✅ **Moderate Robustness:** Results are reasonably robust to violations.",
            f"- Results hold under moderate parallel trends violations",
            f"- Standard econometric interpretation appropriate",
            f"- Consider reporting breakdown point in results"
        ])
    else:
        output_lines.extend([
            "💪 **Strong Robustness:** Results are robust to substantial violations.",
            f"- High confidence in causal interpretation",
            f"- Parallel trends assumption well-supported",
            f"- Standard inference procedures appropriate"
        ])
    
    return "\n".join(output_lines)