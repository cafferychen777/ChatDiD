"""
5-Step DID Workflow Implementation
Following Roth et al. (2023) best practices
"""

import logging
import sys
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging to stderr ONLY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class DiDWorkflow:
    """
    Implements the 5-step DID workflow from the ROADMAP.
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.workflow_state = {
            "step": 0,
            "staggered": None,
            "diagnostics": None,
            "estimation": None,
            "parallel_trends": None,
            "inference": None
        }
    
    async def step1_check_staggered_treatment(
        self, 
        unit_col: str,
        time_col: str,
        treatment_col: str,
        cohort_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Step 1: Determine if treatment is staggered across units.
        
        Returns:
            Dict with:
            - is_staggered: bool
            - treatment_timing: analysis of when units get treated
            - recommendation: "proceed_to_step4" or "proceed_to_step2"
        """
        logger.info("Step 1: Checking for staggered treatment adoption")
        
        if self.analyzer.data is None:
            return {
                "status": "error",
                "message": "No data loaded. Please load data first."
            }
        
        data = self.analyzer.data
        
        # Analyze treatment timing
        if cohort_col and cohort_col in data.columns:
            # Use provided cohort variable
            cohorts = data[cohort_col].dropna().unique()
            n_cohorts = len(cohorts)
            is_staggered = n_cohorts > 1
            
            cohort_summary = data.groupby(cohort_col)[treatment_col].agg(['mean', 'min', 'max'])
            
        else:
            # Infer from treatment variable
            treatment_by_unit_time = data.groupby([unit_col, time_col])[treatment_col].first().reset_index()
            
            # Find first treatment time for each unit
            first_treated = treatment_by_unit_time[treatment_by_unit_time[treatment_col] == 1].groupby(unit_col)[time_col].min()
            
            # Check if all units are treated at the same time
            unique_treatment_times = first_treated.dropna().unique()
            n_cohorts = len(unique_treatment_times)
            is_staggered = n_cohorts > 1
            
            cohort_summary = pd.DataFrame({
                'cohort': unique_treatment_times,
                'n_units': [sum(first_treated == t) for t in unique_treatment_times]
            })
        
        # Store in workflow state
        self.workflow_state["step"] = 1
        self.workflow_state["staggered"] = is_staggered
        
        result = {
            "status": "success",
            "is_staggered": is_staggered,
            "n_cohorts": n_cohorts,
            "cohort_summary": cohort_summary.to_dict() if isinstance(cohort_summary, pd.DataFrame) else cohort_summary,
            "recommendation": "proceed_to_step2" if is_staggered else "proceed_to_step4"
        }
        
        # Build response message
        if is_staggered:
            result["message"] = f"""
## Step 1 Result: Staggered Treatment Detected ⚠️

- **Number of treatment cohorts:** {n_cohorts}
- **Treatment type:** Staggered adoption
- **Implication:** Standard TWFE is likely biased

### Recommendation: Proceed to Step 2
Standard TWFE estimator will produce biased estimates due to forbidden comparisons.
You must use heterogeneity-robust estimators.

**Next:** Run Step 2 - Diagnose TWFE bias
"""
        else:
            result["message"] = f"""
## Step 1 Result: Non-Staggered Treatment ✅

- **Treatment type:** Single adoption time or canonical 2×2
- **Implication:** Standard TWFE may be valid

### Recommendation: Proceed to Step 4
TWFE is not problematic for non-staggered treatments.
You can skip diagnostic steps and proceed to parallel trends assessment.

**Next:** Run Step 4 - Assess parallel trends
"""
        
        return result
    
    async def step2_diagnose_twfe_complete(self) -> Dict[str, Any]:
        """
        Step 2: Run full TWFE diagnostic suite.
        - Goodman-Bacon decomposition
        - Negative weights analysis
        """
        logger.info("Step 2: Running TWFE diagnostics")
        
        if self.workflow_state["step"] < 1:
            return {
                "status": "error",
                "message": "Please run Step 1 first"
            }
        
        if not self.workflow_state["staggered"]:
            return {
                "status": "info",
                "message": "Treatment is not staggered. TWFE diagnostics not needed."
            }
        
        # This will call the actual R implementations
        diagnostics = await self.analyzer.diagnose_twfe(
            run_bacon_decomp=True,
            run_twfe_weights=True
        )
        
        self.workflow_state["step"] = 2
        self.workflow_state["diagnostics"] = diagnostics
        
        return diagnostics
    
    async def step3_apply_robust_estimator(
        self,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Step 3: Apply appropriate heterogeneity-robust estimator.

        Following Roth et al. (2023) best practices, this step:
        1. Runs primary robust estimator
        2. Runs secondary estimator for robustness check
        3. Compares results across methods

        Methods:
        - "auto": Automatically select based on diagnostics
        - "callaway_santanna": CS estimator
        - "sun_abraham": SA estimator
        - "imputation_bjs": BJS imputation
        - "gardner": Gardner two-stage
        - "dcdh": de Chaisemartin & D'Haultfoeuille estimator
        - "efficient": Roth & Sant'Anna efficient estimator
        """
        logger.info(f"Step 3: Applying robust estimator (method={method})")

        if self.workflow_state["step"] < 2 and self.workflow_state["staggered"]:
            return {
                "status": "error",
                "message": "Please run Step 2 (diagnostics) first"
            }

        # Auto-select method based on diagnostics
        if method == "auto":
            if self.workflow_state["diagnostics"]:
                # Logic to select best method based on diagnostic results
                forbidden_weight = self.workflow_state["diagnostics"].get("bacon_decomp", {}).get("forbidden_comparison_weight", 0)
                negative_weights = self.workflow_state["diagnostics"].get("twfe_weights", {}).get("negative_weight_share", 0)

                if forbidden_weight > 0.3 or negative_weights > 0.2:
                    method = "callaway_santanna"  # Most robust
                elif forbidden_weight > 0.1:
                    method = "sun_abraham"  # Fast and convenient
                else:
                    method = "efficient"  # Roth & Sant'Anna efficient estimator
            else:
                method = "callaway_santanna"  # Safe default

        # Define robustness check method pairing (following best practices)
        robustness_pairs = {
            "callaway_santanna": "sun_abraham",
            "sun_abraham": "callaway_santanna",
            "imputation_bjs": "efficient",
            "efficient": "imputation_bjs",
            "gardner": "sun_abraham",
            "dcdh": "callaway_santanna"
        }

        # Step 3a: Apply primary estimator
        logger.info(f"Step 3a: Running primary estimator - {method}")
        primary_estimation = await self.analyzer.estimate_did(method=method)

        # Store primary results
        if primary_estimation.get("status") == "success":
            self.analyzer.results[f"workflow_{method}"] = primary_estimation
            self.analyzer.results["latest"] = primary_estimation
            logger.info(f"Stored primary estimation results: workflow_{method}")
        else:
            logger.warning(f"Primary estimation failed: {primary_estimation.get('message', 'Unknown error')}")
            self.workflow_state["step"] = 3
            self.workflow_state["estimation"] = primary_estimation
            return primary_estimation

        # Step 3b: Apply robustness check estimator
        robustness_method = robustness_pairs.get(method, "sun_abraham")
        logger.info(f"Step 3b: Running robustness check estimator - {robustness_method}")

        robustness_estimation = await self.analyzer.estimate_did(method=robustness_method)

        # Store robustness results
        if robustness_estimation.get("status") == "success":
            self.analyzer.results[f"workflow_{robustness_method}"] = robustness_estimation
            logger.info(f"Stored robustness check results: workflow_{robustness_method}")
        else:
            logger.warning(f"Robustness check failed: {robustness_estimation.get('message', 'Unknown error')}")

        # Step 3c: Compare results
        comparison = self._compare_estimators(primary_estimation, robustness_estimation, method, robustness_method)

        # Store combined results
        combined_result = {
            "status": "success",
            "primary_method": method,
            "robustness_method": robustness_method,
            "primary_estimation": primary_estimation,
            "robustness_estimation": robustness_estimation,
            "comparison": comparison
        }

        self.workflow_state["step"] = 3
        self.workflow_state["estimation"] = combined_result

        return combined_result

    def _compare_estimators(
        self,
        primary: Dict[str, Any],
        robustness: Dict[str, Any],
        primary_name: str,
        robustness_name: str
    ) -> Dict[str, Any]:
        """
        Compare results from two different estimators.

        Returns:
            Comparison report with ATT differences and consistency assessment
        """
        comparison = {
            "primary_method": primary_name,
            "robustness_method": robustness_name
        }

        # Compare overall ATT if available
        if primary.get("status") == "success" and robustness.get("status") == "success":
            primary_att = primary.get("overall_att", {})
            robustness_att = robustness.get("overall_att", {})

            if isinstance(primary_att, dict) and isinstance(robustness_att, dict):
                primary_est = primary_att.get("estimate", 0)
                robustness_est = robustness_att.get("estimate", 0)

                comparison["primary_att"] = primary_est
                comparison["robustness_att"] = robustness_est
                comparison["att_difference"] = abs(primary_est - robustness_est)

                # Calculate relative difference if primary estimate is not zero
                if abs(primary_est) > 0.0001:
                    comparison["relative_difference_pct"] = abs(primary_est - robustness_est) / abs(primary_est) * 100
                else:
                    comparison["relative_difference_pct"] = 0

                # Assess consistency (if difference < 20%, considered consistent)
                comparison["consistent"] = comparison.get("relative_difference_pct", 0) < 20

                # Check if confidence intervals overlap
                primary_ci_lower = primary_att.get("ci_lower", primary_est)
                primary_ci_upper = primary_att.get("ci_upper", primary_est)
                robustness_ci_lower = robustness_att.get("ci_lower", robustness_est)
                robustness_ci_upper = robustness_att.get("ci_upper", robustness_est)

                # CIs overlap if one's lower bound is less than the other's upper bound and vice versa
                ci_overlap = (primary_ci_lower <= robustness_ci_upper) and (robustness_ci_lower <= primary_ci_upper)
                comparison["ci_overlap"] = ci_overlap

        return comparison
    
    async def step4_assess_parallel_trends(self) -> Dict[str, Any]:
        """
        Step 4: Formal parallel trends assessment.
        - Power analysis
        - Sensitivity analysis
        - Robust confidence intervals
        """
        logger.info("Step 4: Assessing parallel trends")
        
        if self.workflow_state["step"] < 3 and self.workflow_state["staggered"]:
            return {
                "status": "error",
                "message": "Please complete Step 3 (estimation) first"
            }
        
        # Run parallel trends tests
        assessment = await self.analyzer.assess_parallel_trends()
        
        self.workflow_state["step"] = 4
        self.workflow_state["parallel_trends"] = assessment
        
        return assessment
    
    async def step5_finalize_inference(
        self,
        cluster_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Step 5: Finalize statistical inference.
        - Proper clustering
        - Handle few treated clusters
        - Generate publication-ready output
        """
        logger.info("Step 5: Finalizing inference")
        
        if self.workflow_state["step"] < 4:
            return {
                "status": "error",
                "message": "Please complete Step 4 (parallel trends) first"
            }
        
        # Finalize inference with proper standard errors
        inference = await self.analyzer.finalize_inference(
            cluster_level=cluster_level
        )
        
        self.workflow_state["step"] = 5
        self.workflow_state["inference"] = inference
        
        # Generate complete report
        report = self._generate_final_report()
        
        return {
            "status": "success",
            "inference": inference,
            "report": report,
            "workflow_complete": True
        }
    
    async def run_complete_workflow(
        self,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_col: str,
        cohort_col: Optional[str] = None,
        method: str = "auto",
        cluster_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete 5-step workflow automatically.
        """
        logger.info("Running complete DID workflow")
        
        results = {
            "workflow_steps": []
        }
        
        # Step 1: Check staggered treatment
        step1 = await self.step1_check_staggered_treatment(
            unit_col, time_col, treatment_col, cohort_col
        )
        results["workflow_steps"].append({"step": 1, "result": step1})
        
        if step1["status"] != "success":
            return results
        
        # Step 2: Diagnose if staggered
        if step1["is_staggered"]:
            step2 = await self.step2_diagnose_twfe_complete()
            results["workflow_steps"].append({"step": 2, "result": step2})
        
        # Step 3: Apply estimator
        step3 = await self.step3_apply_robust_estimator(method=method)
        results["workflow_steps"].append({"step": 3, "result": step3})
        
        # Step 4: Assess parallel trends
        step4 = await self.step4_assess_parallel_trends()
        results["workflow_steps"].append({"step": 4, "result": step4})
        
        # Step 5: Finalize inference
        step5 = await self.step5_finalize_inference(cluster_level=cluster_level)
        results["workflow_steps"].append({"step": 5, "result": step5})
        
        results["status"] = "complete"
        results["final_report"] = step5.get("report", "")
        
        return results
    
    def _generate_final_report(self) -> str:
        """Generate a comprehensive final report."""
        report = """
# DID Analysis Complete Report

## Workflow Summary
"""
        # Add details from each step
        for i, step_name in enumerate(["staggered", "diagnostics", "estimation", "parallel_trends", "inference"], 1):
            if self.workflow_state.get(step_name):
                report += f"\n### Step {i}: {'✅ Completed' if self.workflow_state[step_name] else '⏭️ Skipped'}\n"
                
                # Add specific details for important steps
                if step_name == "staggered" and self.workflow_state[step_name]:
                    is_staggered = self.workflow_state.get("staggered", False)
                    report += f"- Treatment Type: {'Staggered' if is_staggered else 'Non-staggered'}\n"
                
                elif step_name == "diagnostics" and self.workflow_state.get("diagnostics"):
                    diag = self.workflow_state["diagnostics"]
                    if "bacon_decomp" in diag:
                        bacon = diag["bacon_decomp"]
                        if "overall_estimate" in bacon:
                            report += f"- TWFE Estimate: {bacon['overall_estimate']:.4f}\n"
                        if "forbidden_comparison_weight" in bacon:
                            report += f"- Forbidden Comparisons: {bacon['forbidden_comparison_weight']*100:.1f}%\n"
                
                elif step_name == "estimation" and self.workflow_state.get("estimation"):
                    est = self.workflow_state["estimation"]
                    if est.get("status") == "success":
                        # Check if this is dual-estimation result (with robustness check)
                        if "primary_method" in est and "robustness_method" in est:
                            # Dual estimation result
                            report += f"- **Primary Method:** {est['primary_method']}\n"
                            report += f"- **Robustness Check Method:** {est['robustness_method']}\n\n"

                            # Primary estimation results
                            primary = est.get("primary_estimation", {})
                            if primary.get("status") == "success":
                                report += f"**Primary Estimator Results:**\n"
                                primary_att = primary.get("overall_att", {})
                                if isinstance(primary_att, dict):
                                    if "estimate" in primary_att:
                                        report += f"- Overall ATT: {primary_att['estimate']:.4f}\n"
                                    if "se" in primary_att:
                                        report += f"- Standard Error: {primary_att['se']:.4f}\n"
                                    if "ci_lower" in primary_att and "ci_upper" in primary_att:
                                        report += f"- 95% CI: [{primary_att['ci_lower']:.4f}, {primary_att['ci_upper']:.4f}]\n"
                                    if "pvalue" in primary_att:
                                        report += f"- P-value: {primary_att['pvalue']:.4f}\n"

                            # Robustness check results
                            robustness = est.get("robustness_estimation", {})
                            if robustness.get("status") == "success":
                                report += f"\n**Robustness Check Results:**\n"
                                robustness_att = robustness.get("overall_att", {})
                                if isinstance(robustness_att, dict):
                                    if "estimate" in robustness_att:
                                        report += f"- Overall ATT: {robustness_att['estimate']:.4f}\n"
                                    if "se" in robustness_att:
                                        report += f"- Standard Error: {robustness_att['se']:.4f}\n"
                                    if "ci_lower" in robustness_att and "ci_upper" in robustness_att:
                                        report += f"- 95% CI: [{robustness_att['ci_lower']:.4f}, {robustness_att['ci_upper']:.4f}]\n"

                            # Comparison results
                            comparison = est.get("comparison", {})
                            if comparison:
                                report += f"\n**Robustness Comparison:**\n"
                                if "att_difference" in comparison:
                                    report += f"- Absolute Difference: {comparison['att_difference']:.4f}\n"
                                if "relative_difference_pct" in comparison:
                                    report += f"- Relative Difference: {comparison['relative_difference_pct']:.2f}%\n"
                                if "consistent" in comparison:
                                    consistency = "✅ Consistent" if comparison["consistent"] else "⚠️ Inconsistent"
                                    report += f"- Consistency: {consistency}\n"
                                if "ci_overlap" in comparison:
                                    ci_status = "✅ Yes" if comparison["ci_overlap"] else "⚠️ No"
                                    report += f"- Confidence Intervals Overlap: {ci_status}\n"
                        else:
                            # Single estimation result (legacy format)
                            report += f"- Method: {est.get('method', 'Unknown')}\n"
                            if "overall_att" in est:
                                att = est["overall_att"]
                                if isinstance(att, dict):
                                    if "estimate" in att:
                                        report += f"- Overall ATT: {att['estimate']:.4f}\n"
                                    if "se" in att:
                                        report += f"- Standard Error: {att['se']:.4f}\n"
                                    if "ci_lower" in att and "ci_upper" in att:
                                        report += f"- 95% CI: [{att['ci_lower']:.4f}, {att['ci_upper']:.4f}]\n"
                                    if "pvalue" in att:
                                        report += f"- P-value: {att['pvalue']:.4f}\n"
                                else:
                                    report += f"- Overall ATT: {est['overall_att']:.4f}\n"
                            if "n_treated" in est:
                                report += f"- Treated Units: {est['n_treated']}\n"
                
                elif step_name == "parallel_trends" and self.workflow_state.get("parallel_trends"):
                    pt = self.workflow_state["parallel_trends"]
                    if "pretrends_test" in pt:
                        test = pt["pretrends_test"]
                        report += f"- Pre-trends Test p-value: {test.get('p_value', 'N/A')}\n"
                    if "power_results" in pt:
                        power = pt["power_results"]
                        if "80%" in power:
                            report += f"- Power at 80%: Min detectable slope = {power['80%'].get('minimal_detectable_slope', 'N/A')}\n"
        
        return report