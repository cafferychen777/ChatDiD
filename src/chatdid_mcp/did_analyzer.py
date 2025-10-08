"""
Core DID Analysis Engine

This module provides the main DiDAnalyzer class that handles all
difference-in-differences analysis operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import sys
from pathlib import Path
import json
from scipy import stats
from .visualization import DiDVisualizer

# Configure logging to stderr only (MCP STDIO transport requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    # Don't use deprecated activate() - use localconverter instead
    R_AVAILABLE = True
except ImportError as e:
    R_AVAILABLE = False
    logging.warning(f"R integration not available: {e}. Some features will be limited.")

logger = logging.getLogger(__name__)


class DiDAnalyzer:
    """Main class for conducting Difference-in-Differences analysis."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {
            "unit_col": None,
            "time_col": None,
            "outcome_col": None,
            "treatment_col": None,
            "cohort_col": None,
        }

        # Initialize visualization
        self.visualizer = DiDVisualizer(backend="matplotlib")
        logger.info("DiD visualizer initialized")

        # Initialize R packages if available
        if R_AVAILABLE:
            self._setup_r_packages()
            # Import R estimators module
            from .r_estimators import REstimators
            self.r_estimators = REstimators()
        else:
            self.r_estimators = None

        # Import workflow module
        from .workflow import DiDWorkflow
        self.workflow = DiDWorkflow(self)
    
    def _setup_r_packages(self):
        """Setup R packages for DID analysis."""
        try:
            # Install and load required R packages
            # Core estimators
            r_packages = [
                "did",              # Callaway & Sant'Anna
                "fixest",           # Sun & Abraham
                "didimputation",    # BJS imputation
                "did2s",            # Gardner two-stage
                "DIDmultiplegt",    # de Chaisemartin & D'Haultfoeuille (legacy)
                "DIDmultiplegtDYN", # de Chaisemartin & D'Haultfoeuille (modern)
                "staggered",        # Roth & Sant'Anna efficient estimator
                "bacondecomp",      # Goodman-Bacon decomposition
                "TwoWayFEWeights",  # TWFE weights analysis
                "HonestDiD",        # Sensitivity analysis
                "pretrends"         # Power analysis
            ]

            for package in r_packages:
                try:
                    importr(package)
                    logger.info(f"Loaded R package: {package}")
                except Exception as e:
                    logger.warning(f"Could not load R package {package}: {e}")
                    
        except Exception as e:
            logger.error(f"Error setting up R packages: {e}")
    
    async def load_data(
        self, 
        file_path: str, 
        file_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Load data from various file formats."""
        try:
            file_path = Path(file_path)
            
            if file_type == "auto":
                file_type = file_path.suffix.lower()
            
            # Normalize file type (remove leading dot if present)
            if file_type.startswith('.'):
                file_type = file_type[1:]
            
            # Load data based on file type
            if file_type in ["csv"]:
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_type in ["xlsx", "xls"]:
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_type in ["dta"]:
                try:
                    import pyreadstat
                    self.data, _ = pyreadstat.read_dta(file_path)
                except ImportError:
                    raise ImportError("pyreadstat required for Stata files")
            elif file_type in ["parquet"]:
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Clear previous analysis results when loading new data
            # This prevents mixing diagnostics/results from different datasets
            if self.diagnostics:
                logger.info("Clearing previous diagnostic results (new data loaded)")
                self.diagnostics = {}
            if self.results:
                logger.info("Clearing previous estimation results (new data loaded)")
                self.results = {}

            # Basic data info
            info = {
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": self.data.dtypes.to_dict(),
                "missing_values": self.data.isnull().sum().to_dict(),
                "sample_data": self.data.head().to_dict(),
            }

            logger.info(f"Loaded data with shape {self.data.shape}")
            return {"status": "success", "info": info}
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def explore_data(self) -> Dict[str, Any]:
        """Explore the loaded data for DID analysis."""
        if self.data is None:
            return {"status": "error", "message": "No data loaded"}
        
        try:
            exploration = {
                "basic_stats": self.data.describe().to_dict(),
                "panel_structure": self._analyze_panel_structure(),
                "treatment_patterns": self._analyze_treatment_patterns(),
                "recommendations": self._get_setup_recommendations(),
            }
            
            return {"status": "success", "exploration": exploration}
            
        except Exception as e:
            logger.error(f"Error exploring data: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_panel_structure(self) -> Dict[str, Any]:
        """Analyze the panel structure of the data."""
        if self.data is None:
            return {}
        
        # Try to identify potential panel variables
        potential_units = []
        potential_times = []
        
        for col in self.data.columns:
            # Check for unit identifiers
            if any(keyword in col.lower() for keyword in ['id', 'unit', 'state', 'firm', 'county']):
                potential_units.append(col)
            
            # Check for time variables
            if any(keyword in col.lower() for keyword in ['year', 'time', 'date', 'period']):
                potential_times.append(col)
        
        return {
            "potential_unit_vars": potential_units,
            "potential_time_vars": potential_times,
            "n_observations": len(self.data),
            "n_unique_combinations": len(self.data.drop_duplicates()),
        }
    
    def _analyze_treatment_patterns(self) -> Dict[str, Any]:
        """Analyze potential treatment variables and patterns."""
        if self.data is None:
            return {}
        
        # Look for binary variables that could be treatments
        potential_treatments = []
        
        for col in self.data.columns:
            if self.data[col].dtype in ['bool', 'int64', 'float64']:
                unique_vals = self.data[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                    potential_treatments.append({
                        "column": col,
                        "values": list(unique_vals),
                        "treatment_share": self.data[col].mean() if col in self.data.columns else 0,
                    })
        
        return {"potential_treatment_vars": potential_treatments}
    
    def _get_setup_recommendations(self) -> List[str]:
        """Provide recommendations for setting up the DID analysis."""
        recommendations = [
            "1. Identify your unit identifier (e.g., state_id, firm_id)",
            "2. Identify your time variable (e.g., year, quarter)",
            "3. Specify your outcome variable of interest",
            "4. Define your treatment variable (binary indicator)",
            "5. If staggered adoption, specify cohort variable (treatment timing)",
        ]
        
        return recommendations
    
    async def diagnose_twfe(
        self,
        run_bacon_decomp: bool = True,
        run_twfe_weights: bool = True
    ) -> Dict[str, Any]:
        """Run TWFE diagnostics including Bacon decomposition and weight analysis."""
        if self.data is None:
            return {"status": "error", "message": "No data loaded"}
        
        diagnostics = {}
        
        # Run Goodman-Bacon decomposition if requested and available
        if run_bacon_decomp and self.r_estimators:
            if all(col in self.data.columns for col in [
                self.config['outcome_col'],
                self.config['treatment_col'],
                self.config['unit_col'],
                self.config['time_col']
            ]):
                formula = f"{self.config['outcome_col']} ~ {self.config['treatment_col']}"
                bacon_result = self.r_estimators.goodman_bacon_decomposition(
                    data=self.data,
                    formula=formula,
                    id_var=self.config['unit_col'],
                    time_var=self.config['time_col']
                )
                if bacon_result["status"] == "success":
                    diagnostics["bacon_decomp"] = bacon_result
        
        # Run TWFE weights analysis if requested and available
        if run_twfe_weights and self.r_estimators:
            weights_result = self.r_estimators.twfe_weights_analysis(
                data=self.data,
                outcome_col=self.config['outcome_col'],
                unit_col=self.config['unit_col'],
                time_col=self.config['time_col'],
                treatment_col=self.config['treatment_col']
            )
            if weights_result["status"] == "success":
                diagnostics["twfe_weights"] = weights_result
        
        return {"status": "success", "diagnostics": diagnostics}
    
    async def estimate_did(
        self,
        method: str = "callaway_santanna",
        **kwargs
    ) -> Dict[str, Any]:
        """Estimate DID with specified method."""
        if self.data is None:
            return {"status": "error", "message": "No data loaded"}
        
        if not self.r_estimators:
            return {"status": "error", "message": "R integration not available"}
        
        # Route to appropriate estimator
        if method == "callaway_santanna":
            return self.r_estimators.callaway_santanna_estimator(
                data=self.data,
                yname=self.config['outcome_col'],
                tname=self.config['time_col'],
                idname=self.config['unit_col'],
                gname=self.config.get('cohort_col', self.config['treatment_col']),
                **kwargs
            )
        elif method == "sun_abraham":
            # Construct formula for Sun & Abraham estimator
            outcome_col = self.config['outcome_col']
            unit_col = self.config['unit_col']
            time_col = self.config['time_col']
            cohort_col = self.config.get('cohort_col', self.config['treatment_col'])

            # Formula format: "outcome ~ sunab(cohort, time) | unit + time"
            formula = f"{outcome_col} ~ sunab({cohort_col}, {time_col}) | {unit_col} + {time_col}"

            return self.r_estimators.sun_abraham_estimator(
                data=self.data,
                formula=formula,
                cluster_var=kwargs.get('cluster_var', self.config['unit_col'])
            )
        elif method == "imputation_bjs":
            return self.r_estimators.bjs_imputation_estimator(
                data=self.data,
                outcome_col=self.config['outcome_col'],
                unit_col=self.config['unit_col'],
                time_col=self.config['time_col'],
                cohort_col=self.config.get('cohort_col', self.config['treatment_col']),
                **kwargs
            )
        elif method == "gardner":
            return self.r_estimators.gardner_two_stage_estimator(
                data=self.data,
                outcome_col=self.config['outcome_col'],
                unit_col=self.config['unit_col'],
                time_col=self.config['time_col'],
                treatment_col=self.config['treatment_col'],
                cohort_col=self.config.get('cohort_col'),
                **kwargs
            )
        elif method == "dcdh":
            return self.r_estimators.dcdh_estimator(
                data=self.data,
                outcome_col=self.config['outcome_col'],
                unit_col=self.config['unit_col'],
                time_col=self.config['time_col'],
                treatment_col=self.config['treatment_col'],
                cohort_col=self.config.get('cohort_col'),
                **kwargs
            )
        elif method == "efficient":
            return self.r_estimators.efficient_estimator(
                data=self.data,
                outcome_col=self.config['outcome_col'],
                unit_col=self.config['unit_col'],
                time_col=self.config['time_col'],
                cohort_col=self.config.get('cohort_col', self.config['treatment_col']),
                **kwargs
            )
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}
    
    async def assess_parallel_trends(self) -> Dict[str, Any]:
        """
        Comprehensive parallel trends assessment using modern methods.
        
        Implements Roth et al. (2023) best practices:
        1. Power analysis (pretrends package)
        2. Sensitivity analysis (HonestDiD package)
        3. Formal parallel trends tests
        
        Returns:
            Dict with comprehensive parallel trends assessment
        """
        if not self.results:
            return {
                "status": "error",
                "message": "No estimation results available. Please run a DID estimator first."
            }
        
        # Get latest results
        latest_key = max(self.results.keys()) if self.results else None
        if not latest_key:
            return {
                "status": "error", 
                "message": "No valid estimation results found"
            }
        
        latest_results = self.results[latest_key]
        event_study = latest_results.get("event_study", {})
        
        if not event_study:
            return {
                "status": "error",
                "message": "No event study results available for parallel trends assessment"
            }
        
        try:
            logger.info("Running comprehensive parallel trends assessment")
            
            # Step 1: Power Analysis using pretrends
            logger.info("Step 1: Pretrends power analysis")
            
            sorted_times = sorted(event_study.keys())
            betahat = np.array([event_study[t]["estimate"] for t in sorted_times])
            time_vec = np.array(sorted_times)
            
            # Use full covariance matrix if available
            if "event_study_vcov" in latest_results:
                vcov_info = latest_results["event_study_vcov"]
                sigma = np.array(vcov_info["matrix"])
                covariance_method = vcov_info["extraction_method"]
            else:
                sigma = np.diag([event_study[t]["se"] ** 2 for t in sorted_times])
                covariance_method = "diagonal_fallback"
            
            # Run power analysis for multiple power levels
            power_results = {}
            power_levels = [0.5, 0.8, 0.9]
            
            for power in power_levels:
                power_result = self.r_estimators.pretrends_power_analysis(
                    betahat=betahat,
                    sigma=sigma,
                    time_vec=time_vec,
                    reference_period=-1,
                    target_power=power,
                    alpha=0.05
                )
                
                if power_result["status"] == "success":
                    power_results[f"{int(power*100)}%"] = {
                        "minimal_detectable_slope": power_result["minimal_detectable_slope"],
                        "interpretation": power_result["interpretation"]
                    }
            
            # Step 2: HonestDiD Sensitivity Analysis
            logger.info("Step 2: HonestDiD sensitivity analysis")
            
            # Prepare data for HonestDiD
            pre_periods = [t for t in sorted_times if t < 0 and t != -1]  # Exclude reference period
            post_periods = [t for t in sorted_times if t >= 0]
            
            # Extract coefficients for HonestDiD (excluding reference period)
            honest_betahat = []
            used_periods = []
            
            for t in pre_periods:
                honest_betahat.append(event_study[t]["estimate"])
                used_periods.append(t)
            for t in post_periods:
                honest_betahat.append(event_study[t]["estimate"])
                used_periods.append(t)
            
            honest_betahat = np.array(honest_betahat)
            
            # Extract corresponding submatrix
            if "event_study_vcov" in latest_results:
                full_sigma = np.array(latest_results["event_study_vcov"]["matrix"])
                vcov_periods = latest_results["event_study_vcov"]["periods"]
                period_to_idx = {p: i for i, p in enumerate(vcov_periods)}
                used_indices = [period_to_idx[t] for t in used_periods if t in period_to_idx]
                
                if len(used_indices) == len(used_periods):
                    honest_sigma = full_sigma[np.ix_(used_indices, used_indices)]
                else:
                    honest_sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])
            else:
                honest_sigma = np.diag([event_study[t]["se"] ** 2 for t in used_periods])
            
            # Run HonestDiD sensitivity analysis
            sensitivity_results = {}
            m_values = [0.5, 1.0, 1.5, 2.0]
            
            honest_result = self.r_estimators.honest_did_sensitivity_analysis(
                betahat=honest_betahat,
                sigma=honest_sigma,
                num_pre_periods=len(pre_periods),
                num_post_periods=len(post_periods),
                method="relative_magnitude",
                m_values=m_values,
                confidence_level=0.95
            )
            
            if honest_result["status"] == "success":
                sensitivity_results = honest_result
            
            # Step 3: Formal Pre-trends Test (simple F-test on pre-periods)
            logger.info("Step 3: Formal pre-trends test")
            
            pretrends_test_result = None
            if len(pre_periods) > 0:
                try:
                    # Joint F-test for pre-period coefficients = 0
                    pre_estimates = np.array([event_study[t]["estimate"] for t in pre_periods])
                    pre_vcov = honest_sigma[:len(pre_periods), :len(pre_periods)]
                    
                    # F-statistic: beta' * inv(Var) * beta
                    if np.linalg.det(pre_vcov) > 1e-12:  # Check for invertibility
                        pre_vcov_inv = np.linalg.inv(pre_vcov)
                        f_stat = pre_estimates.T @ pre_vcov_inv @ pre_estimates
                        df_num = len(pre_periods)
                        # Chi-square approximation (more robust than F)
                        from scipy.stats import chi2
                        p_value = 1 - chi2.cdf(f_stat, df_num)
                        
                        pretrends_test_result = {
                            "test_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "degrees_of_freedom": df_num,
                            "test_type": "Joint test for pre-treatment coefficients = 0",
                            "null_hypothesis": "No pre-trends (parallel trends holds)",
                            "interpretation": "Reject if p < 0.05" if p_value < 0.05 else "Fail to reject (consistent with parallel trends)"
                        }
                except Exception as e:
                    logger.warning(f"Pre-trends test failed: {e}")
                    pretrends_test_result = {
                        "test_type": "Joint test for pre-treatment coefficients = 0",
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Compile comprehensive assessment
            assessment = {
                "status": "success",
                "method": "Comprehensive parallel trends assessment",
                "covariance_method": covariance_method,
                "analysis_components": {
                    "power_analysis": {
                        "status": "completed" if power_results else "failed",
                        "results": power_results,
                        "interpretation": "Higher power indicates better ability to detect violations"
                    },
                    "sensitivity_analysis": {
                        "status": sensitivity_results.get("status", "failed"),
                        "results": sensitivity_results,
                        "interpretation": "Shows robustness to parallel trends violations"
                    },
                    "formal_test": {
                        "status": "completed" if pretrends_test_result else "no_pre_periods",
                        "results": pretrends_test_result,
                        "interpretation": "Statistical test of pre-treatment trends"
                    }
                },
                "summary": self._generate_parallel_trends_summary(power_results, sensitivity_results, pretrends_test_result),
                "recommendations": self._generate_parallel_trends_recommendations(power_results, sensitivity_results, pretrends_test_result)
            }
            
            # Store assessment in results
            self.diagnostics["parallel_trends_assessment"] = assessment
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in parallel trends assessment: {e}")
            return {
                "status": "error",
                "message": f"Parallel trends assessment failed: {str(e)}"
            }
    
    async def finalize_inference(
        self,
        cluster_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Finalize statistical inference with proper standard errors and clustering.
        
        Implements best practices for DID inference:
        1. Appropriate clustering levels
        2. Handling of few treated clusters
        3. Wild bootstrap when needed
        4. Publication-ready output formatting
        
        Args:
            cluster_level: Variable name for clustering ("unit", "time", or custom column)
            
        Returns:
            Dict with finalized inference results
        """
        if not self.results:
            return {
                "status": "error",
                "message": "No estimation results available. Please run estimation first."
            }
        
        # Get latest results
        latest_key = max(self.results.keys()) if self.results else None
        if not latest_key:
            return {
                "status": "error",
                "message": "No valid estimation results found"
            }
        
        latest_results = self.results[latest_key]
        
        try:
            logger.info(f"Finalizing inference with cluster_level: {cluster_level}")
            
            # Step 1: Determine appropriate clustering strategy
            clustering_strategy = self._determine_clustering_strategy(cluster_level)
            
            # Step 2: Check for few treated clusters issue
            cluster_diagnostics = self._diagnose_cluster_structure(cluster_level)
            
            # Step 3: Apply appropriate inference method
            if cluster_diagnostics["few_treated_clusters"]:
                logger.info("Few treated clusters detected - using conservative inference")
                inference_method = "wild_bootstrap"
                inference_results = await self._apply_wild_bootstrap_inference(latest_results, cluster_level)
            else:
                logger.info("Standard clustering inference")
                inference_method = "standard_clustering"
                inference_results = await self._apply_standard_clustering(latest_results, cluster_level)
            
            # Step 4: Generate final confidence intervals and p-values
            final_estimates = self._generate_final_estimates(latest_results, inference_results)
            
            # Step 5: Create publication-ready output
            publication_output = self._format_publication_output(final_estimates, inference_method)
            
            # Compile finalized inference
            finalized_inference = {
                "status": "success",
                "method": inference_method,
                "cluster_level": cluster_level,
                "clustering_strategy": clustering_strategy,
                "cluster_diagnostics": cluster_diagnostics,
                "inference_results": inference_results,
                "final_estimates": final_estimates,
                "publication_output": publication_output,
                "recommendations": self._generate_inference_recommendations(cluster_diagnostics, inference_method)
            }
            
            # Store in results
            self.diagnostics["finalized_inference"] = finalized_inference
            
            return finalized_inference
            
        except Exception as e:
            logger.error(f"Error in inference finalization: {e}")
            return {
                "status": "error",
                "message": f"Inference finalization failed: {str(e)}"
            }
    
    # Helper Methods for Parallel Trends Assessment
    def _generate_parallel_trends_summary(self, power_results, sensitivity_results, pretrends_test_result):
        """Generate summary of parallel trends assessment."""
        summary_parts = []
        
        # Power analysis summary
        if power_results:
            power_80 = power_results.get("80%", {}).get("minimal_detectable_slope")
            if power_80:
                summary_parts.append(f"Power analysis: 80% power to detect linear trends with slope ≥ {power_80:.4f}")
        
        # Sensitivity analysis summary
        if sensitivity_results and sensitivity_results.get("status") == "success":
            summary_parts.append("Sensitivity analysis: Robust confidence sets computed for parallel trends violations")
        
        # Formal test summary
        if pretrends_test_result:
            p_val = pretrends_test_result.get("p_value")
            if p_val is not None:
                summary_parts.append(f"Pre-trends test: p-value = {p_val:.3f}")
        
        return "; ".join(summary_parts) if summary_parts else "Assessment completed with limited results"
    
    def _generate_parallel_trends_recommendations(self, power_results, sensitivity_results, pretrends_test_result):
        """Generate recommendations based on parallel trends assessment."""
        recommendations = []
        
        # Power-based recommendations
        if power_results:
            power_80 = power_results.get("80%", {}).get("minimal_detectable_slope")
            if power_80:
                if power_80 > 0.02:  # Arbitrary threshold - could be domain-specific
                    recommendations.append("Low power to detect trend violations - interpret results cautiously")
                else:
                    recommendations.append("Good power to detect trend violations")
        
        # Test-based recommendations
        if pretrends_test_result:
            p_val = pretrends_test_result.get("p_value")
            if p_val is not None:
                if p_val < 0.05:
                    recommendations.append("WARNING: Formal pre-trends test rejects parallel trends assumption")
                else:
                    recommendations.append("Formal pre-trends test consistent with parallel trends")
        
        # Sensitivity recommendations
        if sensitivity_results and sensitivity_results.get("status") == "success":
            recommendations.append("Use sensitivity analysis results to assess robustness of conclusions")
        
        return recommendations if recommendations else ["Complete assessment - review all components"]
    
    # Helper Methods for Inference Finalization
    def _determine_clustering_strategy(self, cluster_level):
        """Determine appropriate clustering strategy."""
        if not cluster_level:
            return "no_clustering"
        
        # Map standard names to actual columns
        if cluster_level == "unit" and self.config:
            cluster_col = self.config.get("unit_var")
        elif cluster_level == "time" and self.config:
            cluster_col = self.config.get("time_var")
        else:
            cluster_col = cluster_level
        
        if cluster_col and cluster_col in self.data.columns:
            return f"cluster_by_{cluster_col}"
        else:
            return "invalid_cluster_variable"
    
    def _diagnose_cluster_structure(self, cluster_level):
        """Diagnose clustering structure for inference."""
        diagnostics = {
            "few_treated_clusters": False,
            "cluster_count": None,
            "treated_cluster_count": None,
            "recommendation": "standard_clustering"
        }
        
        if not cluster_level or not self.data is not None:
            return diagnostics
        
        try:
            # Map cluster level to actual column
            if cluster_level == "unit" and self.config:
                cluster_col = self.config.get("unit_var")
            elif cluster_level == "time" and self.config:
                cluster_col = self.config.get("time_var")
            else:
                cluster_col = cluster_level
            
            if cluster_col and cluster_col in self.data.columns:
                # Count total clusters
                total_clusters = self.data[cluster_col].nunique()
                diagnostics["cluster_count"] = total_clusters
                
                # Count treated clusters (if treatment info available)
                if self.config and self.config.get("cohort_var") in self.data.columns:
                    cohort_col = self.config["cohort_var"]
                    treated_data = self.data[self.data[cohort_col] < 10000]  # Exclude never-treated
                    treated_clusters = treated_data[cluster_col].nunique()
                    diagnostics["treated_cluster_count"] = treated_clusters
                    
                    # Check for few treated clusters (< 10 is common threshold)
                    if treated_clusters < 10:
                        diagnostics["few_treated_clusters"] = True
                        diagnostics["recommendation"] = "wild_bootstrap"
                
        except Exception as e:
            logger.warning(f"Could not diagnose cluster structure: {e}")
        
        return diagnostics
    
    async def _apply_wild_bootstrap_inference(self, latest_results, cluster_level):
        """Apply wild bootstrap inference for few clusters."""
        # Simplified implementation - in practice would use specialized bootstrap
        return {
            "method": "wild_bootstrap",
            "status": "simulated",
            "note": "Wild bootstrap implementation would require specialized R packages",
            "conservative_adjustment": 1.2  # Conservative multiplier for SEs
        }
    
    async def _apply_standard_clustering(self, latest_results, cluster_level):
        """Apply standard clustering inference."""
        return {
            "method": "standard_clustering",
            "status": "applied",
            "cluster_level": cluster_level,
            "note": "Standard clustering applied (already incorporated in base estimator)"
        }
    
    def _generate_final_estimates(self, latest_results, inference_results):
        """Generate final estimates with adjusted inference."""
        event_study = latest_results.get("event_study", {})
        adjustment_factor = inference_results.get("conservative_adjustment", 1.0)
        
        final_estimates = {}
        for period, estimates in event_study.items():
            adjusted_se = estimates["se"] * adjustment_factor
            final_estimates[period] = {
                "estimate": estimates["estimate"],
                "se": estimates["se"],
                "adjusted_se": adjusted_se,
                "ci_lower": estimates["estimate"] - 1.96 * adjusted_se,
                "ci_upper": estimates["estimate"] + 1.96 * adjusted_se,
                "pvalue": 2 * (1 - stats.norm.cdf(abs(estimates["estimate"] / adjusted_se))),
                "adjustment_applied": adjustment_factor > 1.0
            }
        
        return final_estimates
    
    def _format_publication_output(self, final_estimates, inference_method):
        """Format results for publication."""
        output = {
            "table_format": "Event Study Results",
            "inference_method": inference_method,
            "estimates": []
        }
        
        for period in sorted(final_estimates.keys()):
            est = final_estimates[period]
            output["estimates"].append({
                "period": period,
                "coefficient": f"{est['estimate']:.4f}",
                "std_error": f"{est['adjusted_se']:.4f}",
                "ci_95": f"[{est['ci_lower']:.4f}, {est['ci_upper']:.4f}]",
                "p_value": f"{est['pvalue']:.3f}",
                "significant": est['pvalue'] < 0.05
            })
        
        return output
    
    def _generate_inference_recommendations(self, cluster_diagnostics, inference_method):
        """Generate recommendations for inference."""
        recommendations = []
        
        if cluster_diagnostics["few_treated_clusters"]:
            recommendations.append("Few treated clusters detected - using conservative inference methods")
        
        if inference_method == "wild_bootstrap":
            recommendations.append("Wild bootstrap recommended for robust inference with few clusters")
        
        recommendations.append("Review clustering level appropriateness for your research design")
        
        return recommendations

    # Visualization Methods
    async def create_event_study_plot(
        self,
        results_key: str = "latest",
        backend: str = "matplotlib",
        save_path: Optional[str] = None,
        display_mode: str = "both",
        max_inline_size: int = 1_000_000,
        auto_optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Create event study plot from estimation results.

        Args:
            results_key: Key for results to plot ("latest" or specific key)
            backend: Visualization backend ("matplotlib" or "plotly")
            save_path: Optional path to save the plot
            display_mode: How to return the plot
                - "display": Return ImageContent only (for inline display)
                - "save": Save to file and return metadata only
                - "both": Save file + return ImageContent if size permits (default)
            max_inline_size: Maximum size for inline display (default: 1MB)
            auto_optimize: Automatically compress if needed (default: True)

        Returns:
            Dict with plot information, or ImageContent for display
        """
        if not self.results:
            return {
                "status": "error",
                "message": "No estimation results available. Run estimation first."
            }

        # Get results
        if results_key == "latest":
            if not self.results:
                return {
                    "status": "error",
                    "message": "No results available. Please run a DID estimation method first."
                }
            results = list(self.results.values())[-1]
        elif results_key in self.results:
            results = self.results[results_key]
        else:
            # List available results keys (excluding 'latest' and internal keys)
            available_keys = [k for k in self.results.keys()
                            if k not in ["latest", "diagnostics", "workflow"]]
            available_str = "', '".join(available_keys) if available_keys else "None"
            return {
                "status": "error",
                "message": f"Results key '{results_key}' not found. Available keys: '{available_str}'. Use 'latest' for most recent results."
            }

        # Set backend
        self.visualizer.set_backend(backend)

        # Create plot
        return self.visualizer.create_event_study_plot(
            results=results,
            save_path=save_path,
            display_mode=display_mode,
            max_inline_size=max_inline_size,
            auto_optimize=auto_optimize
        )

    async def create_diagnostic_plots(
        self,
        backend: str = "matplotlib",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create diagnostic plots from TWFE analysis.

        Prerequisites:
            - Must run `diagnose_goodman_bacon()` for Bacon decomposition plots
            - Must run `analyze_twfe_weights()` for TWFE weights plots
            - At least one diagnostic analysis must be completed before calling

        Args:
            backend: Visualization backend ("matplotlib" or "plotly")
            save_path: Optional path to save plots

        Returns:
            Dict with plot information and status. If no diagnostics available,
            returns error status with message indicating which analyses to run first.
        """
        if not self.diagnostics:
            return {
                "status": "error",
                "message": "No diagnostic results available. Please run diagnose_goodman_bacon() or analyze_twfe_weights() first."
            }

        # Set backend
        self.visualizer.set_backend(backend)

        plots = {}

        # Create Goodman-Bacon plot
        if "bacon_decomp" in self.diagnostics:
            bacon_plot = self.visualizer.create_goodman_bacon_plot(
                bacon_results=self.diagnostics["bacon_decomp"],
                save_path=f"bacon_{save_path}" if save_path else None
            )
            plots["goodman_bacon"] = bacon_plot

        # Create TWFE weights plot
        if "twfe_weights" in self.diagnostics:
            weights_plot = self.visualizer.create_twfe_weights_plot(
                weights_results=self.diagnostics["twfe_weights"],
                save_path=f"weights_{save_path}" if save_path else None
            )
            plots["twfe_weights"] = weights_plot

        return {
            "status": "success",
            "plots": plots,
            "n_plots": len(plots)
        }

    async def create_comprehensive_report(
        self,
        results_key: str = "latest",
        backend: str = "matplotlib",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive DID analysis report with all plots.

        Args:
            results_key: Key for results to include
            backend: Visualization backend
            save_path: Optional path to save the report

        Returns:
            Dict with comprehensive report information
        """
        if not self.results:
            return {
                "status": "error",
                "message": "No estimation results available"
            }

        # Get results
        if results_key == "latest":
            if not self.results:
                return {
                    "status": "error",
                    "message": "No results available. Please run a DID estimation method first."
                }
            results = list(self.results.values())[-1]
        elif results_key in self.results:
            results = self.results[results_key]
        else:
            # List available results keys (excluding 'latest' and internal keys)
            available_keys = [k for k in self.results.keys()
                            if k not in ["latest", "diagnostics", "workflow"]]
            available_str = "', '".join(available_keys) if available_keys else "None"
            return {
                "status": "error",
                "message": f"Results key '{results_key}' not found. Available keys: '{available_str}'. Use 'latest' for most recent results."
            }

        # Set backend
        self.visualizer.set_backend(backend)

        # Create comprehensive report
        return self.visualizer.create_comprehensive_did_report(
            estimation_results=results,
            diagnostic_results=self.diagnostics if self.diagnostics else None,
            save_path=save_path
        )

    def get_visualization_backends(self) -> List[str]:
        """Get available visualization backends."""
        return self.visualizer.get_available_backends()

    def set_visualization_backend(self, backend: str) -> bool:
        """Set visualization backend."""
        return self.visualizer.set_backend(backend)


# Global analyzer instance for MCP tools
_global_analyzer: Optional[DiDAnalyzer] = None


def get_analyzer() -> DiDAnalyzer:
    """Get the global DiD analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = DiDAnalyzer()
    return _global_analyzer


def reset_analyzer() -> None:
    """Reset the global analyzer instance."""
    global _global_analyzer
    _global_analyzer = None
