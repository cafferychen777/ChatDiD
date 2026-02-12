"""
5-Step DID Workflow Implementation
Following Roth et al. (2023) best practices
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DiDWorkflow:
    """
    Implements the 5-step DID workflow from Roth et al. (2023).
    Enhanced with severity bands, evidence assessment, and interpretation
    following the DID Analysis Skill framework.
    """

    # Severity bands for TWFE diagnostics (from DID skill).
    # Ordered from most to least severe — classify_severity iterates these.
    SEVERITY_THRESHOLDS = [
        ("SEVERE",   0.50),   # >50%
        ("MODERATE", 0.25),   # 25-50%
        ("MILD",     0.10),   # 10-25%
    ]
    SEVERITY_DEFAULT = "MINIMAL"  # <10%

    # Breakdown M interpretation bands.
    # Ordered from strongest to weakest — interpret_breakdown_m iterates these.
    BREAKDOWN_M_THRESHOLDS = [
        ("FAIRLY_ROBUST", 1.5),   # >1.5
        ("MODERATE",      1.0),   # 1.0-1.5
    ]
    BREAKDOWN_M_DEFAULT = "WEAK"  # <1.0

    # Estimator comparison: relative difference threshold for agreement
    COMPARISON_AGREEMENT_PCT = 50  # >50% → estimators disagree

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.reset()

    def reset(self):
        """Reset workflow state.  Called when new data is loaded."""
        self.workflow_state = {
            "step": 0,
            "staggered": None,
            "diagnostics": None,
            "estimation": None,
            "parallel_trends": None,
            "inference": None,
        }

    @classmethod
    def classify_severity(cls, value: float) -> str:
        """Classify a diagnostic metric into severity bands."""
        for label, threshold in cls.SEVERITY_THRESHOLDS:
            if value > threshold:
                return label
        return cls.SEVERITY_DEFAULT

    @classmethod
    def interpret_breakdown_m(cls, breakdown_m) -> dict:
        """Interpret breakdown M value from HonestDiD sensitivity analysis.

        IMPORTANT: None → "UNKNOWN" (safe default).  Callers that know the
        sensitivity analysis ran successfully and produced no breakdown point
        should treat that case as "STRONG" *before* calling this method.
        """
        interpretations = {
            "STRONG": "Effect robust to all tested M values",
            "FAIRLY_ROBUST": "Post-treatment violations must be substantially larger than pre-treatment to invalidate",
            "MODERATE": "Robust if post-treatment violations similar to pre-treatment violations",
            "WEAK": "Effect fragile; even smaller-than-pre violations can invalidate",
            "UNKNOWN": "Sensitivity analysis did not produce a breakdown value",
        }

        if breakdown_m is None:
            strength = "UNKNOWN"
        else:
            strength = cls.BREAKDOWN_M_DEFAULT
            for label, threshold in cls.BREAKDOWN_M_THRESHOLDS:
                if breakdown_m > threshold:
                    strength = label
                    break

        return {
            "value": breakdown_m,
            "strength": strength,
            "interpretation": interpretations[strength]
        }

    def assess_evidence(self) -> dict:
        """
        Synthesize all workflow steps into a final evidence assessment.
        Following the DID skill's evidence assessment framework.

        Returns one of: STRONG EVIDENCE, SUGGESTIVE, EVIDENCE OF NULL,
        UNINFORMATIVE, FRAGILE, or MIXED.
        """
        estimation = self.workflow_state.get("estimation", {})
        parallel_trends = self.workflow_state.get("parallel_trends", {})

        # Extract key indicators
        att_significant = False
        att_estimate = None
        estimators_agree = True
        pre_test_pass = True
        power_quality = "UNKNOWN"
        breakdown_strength = "UNKNOWN"

        # From estimation (Step 3)
        if estimation and estimation.get("status") == "success":
            primary = estimation.get("primary_estimation", estimation)
            att_info = primary.get("overall_att", {})
            if isinstance(att_info, dict):
                att_estimate = att_info.get("estimate")
                pval = att_info.get("pvalue")
                if pval is not None:
                    att_significant = pval < 0.05

            comparison = estimation.get("comparison", {})
            if comparison:
                rel_diff = comparison.get("relative_difference_pct", 0)
                if rel_diff > self.COMPARISON_AGREEMENT_PCT:
                    estimators_agree = False

        # From parallel trends (Step 4)
        # assess_parallel_trends returns nested analysis_components structure:
        #   {"analysis_components": {"formal_test": {"results": ...}, ...}}
        if parallel_trends and parallel_trends.get("status") == "success":
            components = parallel_trends.get("analysis_components", {})

            formal_test = components.get("formal_test", {}).get("results") or {}
            if formal_test:
                pt_pval = formal_test.get("p_value")
                if pt_pval is not None:
                    pre_test_pass = pt_pval > 0.05

            sensitivity = components.get("sensitivity_analysis", {}).get("results") or {}
            if sensitivity and sensitivity.get("status") == "success":
                bm = sensitivity.get("breakdown_point")
                if bm is None:
                    # Analysis ran successfully but no M value caused breakdown
                    # → effect is genuinely robust to all tested M values.
                    breakdown_strength = "STRONG"
                else:
                    breakdown_strength = self.interpret_breakdown_m(bm).get("strength", "UNKNOWN")

        # Determine verdict
        if att_significant and estimators_agree and breakdown_strength in ("STRONG", "FAIRLY_ROBUST"):
            verdict = "STRONG EVIDENCE"
            summary = "Significant effect, robust across estimators and sensitivity analysis"
        elif att_significant and estimators_agree and breakdown_strength == "MODERATE":
            verdict = "SUGGESTIVE"
            summary = "Significant effect with moderate robustness to parallel trends violations"
        elif not att_significant and estimators_agree:
            verdict = "EVIDENCE OF NULL"
            summary = "No significant effect detected, consistent across estimators"
        elif att_significant and breakdown_strength == "WEAK":
            verdict = "FRAGILE"
            summary = "Significant effect but fragile to parallel trends violations"
        elif not estimators_agree:
            verdict = "MIXED"
            summary = "Estimators disagree substantially; results depend on modeling assumptions"
        else:
            verdict = "UNINFORMATIVE"
            summary = "Insufficient power or information to draw conclusions"

        return {
            "verdict": verdict,
            "summary": summary,
            "details": {
                "att_significant": att_significant,
                "att_estimate": att_estimate,
                "estimators_agree": estimators_agree,
                "pre_test_pass": pre_test_pass,
                "power_quality": power_quality,
                "breakdown_strength": breakdown_strength
            }
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

        # Preprocessing: delegate to the single source of truth
        processed = self.analyzer.prepare_data_for_estimation(
            unit_col=unit_col,
            time_col=time_col,
            treatment_col=cohort_col if (cohort_col and cohort_col in data.columns) else treatment_col,
            cohort_col=cohort_col
        )
        actual_cohort_col = processed['cohort_col']
        unit_col = processed['unit_col']

        # Analyze treatment timing using the resolved cohort column
        cohort_values = self.analyzer.data[actual_cohort_col]
        treated_cohorts = cohort_values[cohort_values > 0].dropna().unique()
        n_cohorts = len(treated_cohorts)
        is_staggered = n_cohorts > 1

        # Count distinct units per cohort
        treated_data = self.analyzer.data[self.analyzer.data[actual_cohort_col] > 0]
        if len(treated_data) > 0:
            cohort_summary = treated_data.groupby(actual_cohort_col)[unit_col].nunique().reset_index()
            cohort_summary.columns = ['cohort', 'n_units']
        else:
            cohort_summary = pd.DataFrame({'cohort': [], 'n_units': []})

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
            result["message"] = (
                "Step 1 Result: Staggered Treatment Detected\n\n"
                f"- Number of treatment cohorts: {n_cohorts}\n"
                "- Treatment type: Staggered adoption\n"
                "- Implication: Standard TWFE is likely biased\n\n"
                "Recommendation: Proceed to Step 2\n"
                "Standard TWFE estimator will produce biased estimates due to forbidden comparisons.\n"
                "You must use heterogeneity-robust estimators.\n\n"
                "Next: Run Step 2 - Diagnose TWFE bias"
            )
        else:
            result["message"] = (
                "Step 1 Result: Non-Staggered Treatment\n\n"
                "- Treatment type: Single adoption time or canonical 2x2\n"
                "- Implication: Standard TWFE may be valid\n\n"
                "Recommendation: Proceed to Step 4\n"
                "TWFE is not problematic for non-staggered treatments.\n"
                "You can skip diagnostic steps and proceed to parallel trends assessment.\n\n"
                "Next: Run Step 4 - Assess parallel trends"
            )
        
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
        result = await self.analyzer.diagnose_twfe(
            run_bacon_decomp=True,
            run_twfe_weights=True
        )

        # Unwrap: diagnose_twfe returns {"status": ..., "diagnostics": {...}}
        # The actual diagnostic data lives under the "diagnostics" key.
        if result.get("status") != "error":
            diag_data = result.get("diagnostics", {})
            forbidden_weight = diag_data.get("bacon_decomp", {}).get("forbidden_comparison_weight", 0)
            negative_weight = diag_data.get("twfe_weights", {}).get("negative_weight_share", 0)

            forbidden_severity = self.classify_severity(forbidden_weight)
            negative_severity = self.classify_severity(negative_weight)
            overall_severity = forbidden_severity if forbidden_weight > negative_weight else negative_severity

            result["severity_assessment"] = {
                "forbidden_weight_severity": forbidden_severity,
                "negative_weight_severity": negative_severity,
                "overall_severity": overall_severity,
                "forbidden_weight_pct": round(forbidden_weight * 100, 1),
                "negative_weight_pct": round(negative_weight * 100, 1),
                "recommendation": self._severity_recommendation(overall_severity)
            }

        self.workflow_state["step"] = 2
        self.workflow_state["diagnostics"] = result

        return result

    def _severity_recommendation(self, severity: str) -> str:
        """Return recommendation text based on severity band."""
        recommendations = {
            "SEVERE": "Abandon TWFE entirely; use robust estimators (Callaway-Sant'Anna or Sun-Abraham recommended)",
            "MODERATE": "TWFE likely problematic; strongly prefer robust estimators",
            "MILD": "Use TWFE with caution; run robust estimators as robustness check",
            "MINIMAL": "TWFE may be acceptable; robust estimators still recommended as best practice"
        }
        return recommendations.get(severity, "Run robust estimators")
    
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
        - "gsynth": Generalized synthetic control (Xu 2017)
        - "synthdid": Synthetic DID (Arkhangelsky et al. 2019)
        - "drdid": Doubly robust DID (Sant'Anna & Zhao 2020)
        - "etwfe": Extended TWFE (Wooldridge 2021)
        - "efficient": DISABLED (see KNOWN_ISSUES.md)
        """
        logger.info(f"Step 3: Applying robust estimator (method={method})")

        if self.workflow_state["step"] < 2 and self.workflow_state["staggered"]:
            return {
                "status": "error",
                "message": "Please run Step 2 (diagnostics) first"
            }

        # Auto-select method based on diagnostics and severity bands
        if method == "auto":
            if self.workflow_state["diagnostics"]:
                severity = self.workflow_state["diagnostics"].get("severity_assessment", {})
                overall_severity = severity.get("overall_severity", "MINIMAL")

                if overall_severity in ("SEVERE", "MODERATE"):
                    method = "callaway_santanna"  # Most robust for problematic TWFE
                elif overall_severity == "MILD":
                    method = "sun_abraham"  # Fast and convenient
                else:
                    method = "imputation_bjs"  # Efficient when TWFE bias is minimal

                logger.info(f"Auto-selected method '{method}' based on severity: {overall_severity}")
            else:
                method = "callaway_santanna"  # Safe default

        # Define robustness check method pairing (following best practices)
        # Note: "efficient" estimator is DISABLED due to systematic issues (see KNOWN_ISSUES.md)
        #
        # Pairing strategy based on Roth et al. (2023) and recent DID literature:
        # - Within DID family: CS ↔ SA, BJS ↔ Gardner, DCDH ↔ CS
        # - Synthetic control vs DID: gsynth ↔ CS (cross-method validation)
        # - Within synthetic control: synthdid ↔ gsynth
        robustness_pairs = {
            "callaway_santanna": "sun_abraham",
            "sun_abraham": "callaway_santanna",
            "imputation_bjs": "gardner",
            "gardner": "sun_abraham",
            "dcdh": "callaway_santanna",
            "gsynth": "callaway_santanna",
            "synthdid": "gsynth",
            "drdid": "callaway_santanna",
            "etwfe": "callaway_santanna",
        }

        # Step 3a: Apply primary estimator
        logger.info(f"Step 3a: Running primary estimator - {method}")
        primary_estimation = await self.analyzer.estimate_did(method=method)

        # Store primary results.
        # "latest" is set to the PRIMARY estimator intentionally: Steps 4-5
        # read it for parallel trends assessment and final inference.
        # The robustness estimator is for comparison only — it must NOT
        # overwrite "latest".
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

                # Assess consistency using DID skill CV thresholds
                rel_diff = comparison.get("relative_difference_pct", 0)
                if rel_diff < 20:
                    comparison["consistency_assessment"] = "AGREEMENT"
                    comparison["consistency_message"] = "Estimators agree; results are robust to method choice"
                elif rel_diff < 50:
                    comparison["consistency_assessment"] = "INVESTIGATE"
                    comparison["consistency_message"] = "Moderate discrepancy; investigate modeling assumptions"
                else:
                    comparison["consistency_assessment"] = "SERIOUS_CONCERN"
                    comparison["consistency_message"] = "Large discrepancy; results depend on modeling assumptions"
                comparison["consistent"] = rel_diff < 20

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
        
        # Generate evidence assessment
        evidence = self.assess_evidence()

        # Generate complete report
        report = self._generate_final_report()

        return {
            "status": "success",
            "inference": inference,
            "evidence_assessment": evidence,
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
        """Generate a comprehensive final report with severity bands and evidence assessment."""
        report = """
# DID Analysis Complete Report
 Workflow Summary
"""
        # Add details from each step
        for i, step_name in enumerate(["staggered", "diagnostics", "estimation", "parallel_trends", "inference"], 1):
            if self.workflow_state.get(step_name):
                report += f"\nStep {i}: {'Completed' if self.workflow_state[step_name] else 'Skipped'}\n"

                # Add specific details for important steps
                if step_name == "staggered" and self.workflow_state[step_name]:
                    is_staggered = self.workflow_state.get("staggered", False)
                    report += f"- Treatment Type: {'Staggered' if is_staggered else 'Non-staggered'}\n"

                elif step_name == "diagnostics" and self.workflow_state.get("diagnostics"):
                    diag = self.workflow_state["diagnostics"]
                    diag_data = diag.get("diagnostics", {})
                    if "bacon_decomp" in diag_data:
                        bacon = diag_data["bacon_decomp"]
                        if "overall_estimate" in bacon:
                            report += f"- TWFE Estimate: {bacon['overall_estimate']:.4f}\n"
                        if "forbidden_comparison_weight" in bacon:
                            report += f"- Forbidden Comparisons: {bacon['forbidden_comparison_weight']*100:.1f}%\n"

                    # Add severity assessment (from DID skill)
                    severity = diag.get("severity_assessment", {})
                    if severity:
                        report += f"\nTWFE Bias Severity Assessment:\n"
                        report += f"- Forbidden Weight Severity: {severity.get('forbidden_weight_severity', 'N/A')} ({severity.get('forbidden_weight_pct', 'N/A')}%)\n"
                        report += f"- Negative Weight Severity: {severity.get('negative_weight_severity', 'N/A')} ({severity.get('negative_weight_pct', 'N/A')}%)\n"
                        report += f"- Overall Severity: {severity.get('overall_severity', 'N/A')}\n"
                        report += f"- Recommendation: {severity.get('recommendation', 'N/A')}\n"
                
                elif step_name == "estimation" and self.workflow_state.get("estimation"):
                    est = self.workflow_state["estimation"]
                    if est.get("status") == "success":
                        # Check if this is dual-estimation result (with robustness check)
                        if "primary_method" in est and "robustness_method" in est:
                            # Dual estimation result
                            report += f"- Primary Method: {est['primary_method']}\n"
                            report += f"- Robustness Check Method: {est['robustness_method']}\n\n"

                            # Primary estimation results
                            primary = est.get("primary_estimation", {})
                            if primary.get("status") == "success":
                                report += f"Primary Estimator Results:\n"
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
                                report += f"\nRobustness Check Results:\n"
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
                                report += f"\nRobustness Comparison:\n"
                                if "att_difference" in comparison:
                                    report += f"- Absolute Difference: {comparison['att_difference']:.4f}\n"
                                if "relative_difference_pct" in comparison:
                                    report += f"- Relative Difference: {comparison['relative_difference_pct']:.2f}%\n"
                                if "consistent" in comparison:
                                    consistency = "Consistent" if comparison["consistent"] else "Inconsistent"
                                    report += f"- Consistency: {consistency}\n"
                                if "ci_overlap" in comparison:
                                    ci_status = "Yes" if comparison["ci_overlap"] else "No"
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
                    components = pt.get("analysis_components", {})

                    formal_test = components.get("formal_test", {}).get("results") or {}
                    if formal_test:
                        report += f"- Pre-trends Test p-value: {formal_test.get('p_value', 'N/A')}\n"

                    power = components.get("power_analysis", {}).get("results") or {}
                    if power and "80%" in power:
                        report += f"- Power at 80%: Min detectable slope = {power['80%'].get('minimal_detectable_slope', 'N/A')}\n"

                    sensitivity = components.get("sensitivity_analysis", {}).get("results") or {}
                    if sensitivity and sensitivity.get("status") == "success":
                        bp = sensitivity.get("breakdown_point")
                        bp_str = f"M = {bp}" if bp is not None else "None (robust to all tested M)"
                        report += f"- HonestDiD Breakdown Point: {bp_str}\n"

        # Add Evidence Assessment section
        evidence = self.assess_evidence()
        report += f"\nEvidence Assessment\n"
        report += f"Verdict: {evidence['verdict']}\n\n"
        report += f"{evidence['summary']}\n\n"
        report += "Details:\n"
        details = evidence.get("details", {})
        report += f"- ATT Significant: {'Yes' if details.get('att_significant') else 'No'}\n"
        if details.get("att_estimate") is not None:
            report += f"- ATT Estimate: {details['att_estimate']:.4f}\n"
        report += f"- Estimators Agree: {'Yes' if details.get('estimators_agree') else 'No'}\n"
        report += f"- Pre-test Pass: {'Yes' if details.get('pre_test_pass') else 'No'}\n"
        report += f"- Breakdown M Strength: {details.get('breakdown_strength', 'N/A')}\n"

        return report
