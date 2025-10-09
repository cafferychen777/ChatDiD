"""
R-based DID estimators implementation
Implements actual calls to R packages for DID analysis
"""

import logging
import sys
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Configure logging to stderr ONLY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except ImportError as e:
    R_AVAILABLE = False
    logger.error(f"R integration not available: {e}")


class REstimators:
    """
    Implements actual R package calls for DID estimation.
    """
    
    def __init__(self):
        self.r_packages = {}
        if R_AVAILABLE:
            self._load_r_packages()
    
    def _load_r_packages(self):
        """Load required R packages."""
        required_packages = {
            'bacondecomp': 'Goodman-Bacon decomposition',
            'TwoWayFEWeights': 'TWFE weights analysis',
            'did': 'Callaway & Sant\'Anna estimator',
            'fixest': 'Sun & Abraham estimator',
            'didimputation': 'BJS imputation',
            'did2s': 'Gardner two-stage',
            'HonestDiD': 'Sensitivity analysis',
            'pretrends': 'Power analysis',
            'DIDmultiplegt': 'de Chaisemartin & D\'Haultfoeuille estimator (legacy)',
            'DIDmultiplegtDYN': 'de Chaisemartin & D\'Haultfoeuille dynamic estimator (modern)',
            'staggered': 'Roth & Sant\'Anna efficient estimator',
            'gsynth': 'Generalized synthetic control (Xu 2017)',
            'synthdid': 'Synthetic difference-in-differences (Arkhangelsky et al. 2019)'
        }
        
        for pkg_name, description in required_packages.items():
            try:
                self.r_packages[pkg_name] = importr(pkg_name)
                logger.info(f"Loaded R package: {pkg_name} ({description})")
            except Exception as e:
                logger.warning(f"Could not load R package {pkg_name}: {e}")
    
    def goodman_bacon_decomposition(
        self,
        data: pd.DataFrame,
        formula: str,
        id_var: str,
        time_var: str
    ) -> Dict[str, Any]:
        """
        Run Goodman-Bacon (2021) decomposition using bacondecomp::bacon()
        
        Args:
            data: Panel data DataFrame
            formula: R formula (e.g., "outcome ~ treatment")
            id_var: Unit identifier column
            time_var: Time identifier column
            
        Returns:
            Dict with decomposition results
        """
        if not R_AVAILABLE or 'bacondecomp' not in self.r_packages:
            return {
                "status": "error",
                "message": "bacondecomp R package not available"
            }
        
        try:
            logger.info("Running Goodman-Bacon decomposition")

            # Parse formula to extract treatment variable
            import re
            match = re.search(r'~\s*(\w+)', formula)
            if not match:
                return {
                    "status": "error",
                    "message": "Could not parse treatment variable from formula"
                }
            
            treatment_var = match.group(1)
            
            # Prepare data with corrected treatment variable for Goodman-Bacon
            data_corrected = self._prepare_data_for_bacon(data, treatment_var, id_var, time_var)
            
            if data_corrected is None:
                return {
                    "status": "error", 
                    "message": "Data preparation failed: insufficient treatment variation"
                }

            # Convert pandas DataFrame to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data_corrected)

            # Get the bacon function
            bacon = self.r_packages['bacondecomp'].bacon

            # Suppress R output to avoid JSON parsing issues
            robjects.r('sink("/dev/null")')

            try:
                # Run decomposition with corrected data
                # Use the corrected treatment variable in formula
                corrected_formula = formula.replace(treatment_var, 'treat_corrected')
                
                result = bacon(
                    robjects.Formula(corrected_formula),
                    data=r_data,
                    id_var=robjects.StrVector([id_var]),
                    time_var=robjects.StrVector([time_var])
                )
            finally:
                # Restore R output
                robjects.r('sink()')
            
            # Extract results - bacon() returns a data frame directly
            try:
                # The bacon() function returns a data.frame with details
                # Convert directly to pandas
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    decomp_df = robjects.conversion.rpy2py(result)
                
                # Process results - the dataframe has columns: treated, untreated, estimate, weight, type
                comparison_types = {}
                overall_estimate = 0
                forbidden_weight = 0
                
                # Group by type and aggregate
                if 'type' in decomp_df.columns:
                    grouped = decomp_df.groupby('type').agg({
                        'weight': 'sum',
                        'estimate': lambda x: np.average(x, weights=decomp_df.loc[x.index, 'weight'])
                    }).reset_index()
                    
                    for _, row in grouped.iterrows():
                        comp_type = row['type']
                        weight = row['weight']
                        estimate = row['estimate']
                        
                        comparison_types[comp_type] = {
                            'weight': weight,
                            'estimate': estimate
                        }
                        
                        overall_estimate += weight * estimate
                        
                        # Check for forbidden comparisons
                        if 'Later' in str(comp_type) and 'Earlier' in str(comp_type):
                            forbidden_weight += weight
                else:
                    # Fallback if structure is different
                    for _, row in decomp_df.iterrows():
                        comp_type = row.get('type', 'Unknown')
                        weight = row.get('weight', 0)
                        estimate = row.get('estimate', row.get('avg_est', 0))
                        
                        if comp_type not in comparison_types:
                            comparison_types[comp_type] = {'weight': 0, 'estimate': 0}
                        
                        comparison_types[comp_type]['weight'] += weight
                        # Weighted average for estimate
                        comparison_types[comp_type]['estimate'] = estimate
                        
                        overall_estimate += weight * estimate
                        
                        if 'Later' in str(comp_type) and 'Earlier' in str(comp_type):
                            forbidden_weight += weight
                        
            except Exception as e:
                logger.error(f"Error extracting bacon results: {e}")
                # Try to at least get the decomposition DataFrame
                try:
                    # If we can't process it, return the raw dataframe info
                    return {
                        "status": "partial",
                        "message": f"Bacon ran but extraction failed: {e}",
                        "raw_output": str(result)
                    }
                except:
                    raise e
            
            return {
                "status": "success",
                "overall_estimate": overall_estimate,
                "comparison_types": comparison_types,
                "forbidden_comparison_weight": forbidden_weight,
                "decomposition_df": decomp_df.to_dict(),
                "warning": forbidden_weight > 0.1
            }
            
        except Exception as e:
            logger.error(f"Error in Goodman-Bacon decomposition: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _prepare_data_for_bacon(self, data: pd.DataFrame, treatment_var: str, 
                               id_var: str, time_var: str) -> pd.DataFrame:
        """
        Prepare data for Goodman-Bacon decomposition by constructing proper treatment variable.
        
        The original data might have treatment variables that don't follow standard DID convention.
        This method creates a corrected treatment variable that is 1 only after treatment occurs.
        
        Args:
            data: Original panel data
            treatment_var: Name of treatment variable in data
            id_var: Unit identifier column
            time_var: Time identifier column
            
        Returns:
            DataFrame with corrected treatment variable, or None if insufficient variation
        """
        logger.info("Preparing data for Goodman-Bacon decomposition")
        
        data_corrected = data.copy()
        
        # Try to detect if we have a cohort/timing variable
        potential_cohort_vars = ['first.treat', 'cohort', 'treat_year', 'first_treat']
        cohort_var = None
        
        for var in potential_cohort_vars:
            if var in data_corrected.columns:
                cohort_var = var
                break
        
        if cohort_var is not None:
            logger.info(f"Using cohort variable: {cohort_var}")
            
            # Standard DID: treat_corrected should be 1 only for periods >= cohort_var
            # and cohort_var should be > 0 (0 typically means never treated)
            data_corrected['treat_corrected'] = 0
            
            # Set treatment to 1 for units where time >= first treatment time (and first treatment > 0)
            treated_mask = (data_corrected[cohort_var] > 0) & (data_corrected[time_var] >= data_corrected[cohort_var])
            data_corrected.loc[treated_mask, 'treat_corrected'] = 1
            
            # Report statistics
            never_treated = (data_corrected[cohort_var] == 0).sum()
            treated_obs = (data_corrected['treat_corrected'] == 1).sum()
            
            logger.info(f"Corrected treatment variable created:")
            logger.info(f"  - Never treated observations: {never_treated}")
            logger.info(f"  - Treated observations: {treated_obs}")
            logger.info(f"  - Total observations: {len(data_corrected)}")
            
        else:
            logger.warning("No cohort variable found, using original treatment variable")
            
            # Check if original treatment variable has temporal variation within units
            unit_variation = data_corrected.groupby(id_var)[treatment_var].nunique()
            units_with_variation = (unit_variation > 1).sum()
            
            if units_with_variation == 0:
                logger.error("No units have treatment variation over time")
                return None
            
            # Use original treatment variable
            data_corrected['treat_corrected'] = data_corrected[treatment_var]
        
        # Verify we have sufficient variation for Goodman-Bacon decomposition
        treat_counts = data_corrected['treat_corrected'].value_counts()
        
        if len(treat_counts) < 2:
            logger.error("Insufficient treatment variation for decomposition")
            return None
        
        # Check that we have units with temporal treatment variation
        unit_variation = data_corrected.groupby(id_var)['treat_corrected'].nunique()
        units_with_variation = (unit_variation > 1).sum()
        
        if units_with_variation < 2:
            logger.error(f"Only {units_with_variation} units have treatment variation over time")
            logger.error("Goodman-Bacon decomposition requires multiple units with temporal treatment changes")
            return None
        
        logger.info(f"Data preparation successful: {units_with_variation} units with treatment variation")
        return data_corrected
    
    def twfe_weights_analysis(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        treatment_col: str
    ) -> Dict[str, Any]:
        """
        Analyze TWFE weights using TwoWayFEWeights::twowayfeweights()
        
        Returns:
            Dict with weight analysis including negative weights share
        """
        if not R_AVAILABLE or 'TwoWayFEWeights' not in self.r_packages:
            return {
                "status": "error",
                "message": "TwoWayFEWeights R package not available"
            }
        
        try:
            logger.info("Running TWFE weights analysis")
            
            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)
            
            # Get the twowayfeweights function
            twowayfeweights = self.r_packages['TwoWayFEWeights'].twowayfeweights
            
            # Run weights analysis
            # R: weights <- twowayfeweights(data, Y, G, T, D, type="feTR")
            result = twowayfeweights(
                r_data,
                Y=outcome_col,
                G=unit_col,
                T=time_col,
                D=treatment_col,
                type="feTR"
            )
            
            # Extract key statistics
            # The function returns various statistics about weights
            try:
                weights = result.rx2('weights')
                # Convert to numpy array for comparison
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    weights_array = robjects.conversion.rpy2py(weights)
                
                negative_weights_mask = weights_array < 0
                negative_weights = weights_array[negative_weights_mask].sum()
                total_weights = np.abs(weights_array).sum()
                n_negative = negative_weights_mask.sum()
                negative_weight_share = abs(negative_weights) / total_weights if total_weights > 0 else 0
            except:
                # Fallback
                negative_weights = 0
                negative_weight_share = 0
                n_negative = 0
            
            # Robustness measure
            sensitivity = result.rx2('sensitivity_measure')[0] if 'sensitivity_measure' in result.names else None
            
            return {
                "status": "success",
                "negative_weight_share": negative_weight_share,
                "n_negative_weights": int(n_negative),
                "robustness_measure": sensitivity,
                "warning": negative_weight_share > 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in TWFE weights analysis: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def callaway_santanna_estimator(
        self,
        data: pd.DataFrame,
        yname: str,
        tname: str,
        idname: str,
        gname: str,
        control_group: str = "notyettreated",
        xformla: Optional[str] = None,
        anticipation: int = 0,
        clustervars: Optional[str] = None,
        est_method: str = "dr",
        bstrap: bool = True,
        biters: int = 1000,
        alp: float = 0.05,
        cband: bool = True,
        weightsname: Optional[str] = None,
        panel: bool = True,
        allow_unbalanced_panel: bool = False,
        base_period: str = "varying",
        aggregation_type: str = "dynamic"
    ) -> Dict[str, Any]:
        """
        Enhanced Callaway & Sant'Anna (2021) estimator with full parameter support
        
        Args:
            data: Panel data
            yname: Outcome variable name
            tname: Time variable name
            idname: Unit ID variable name
            gname: Group/cohort variable name (first treatment time)
            control_group: "notyettreated" or "nevertreated"
            xformla: Covariate formula (e.g., "~ X1 + X2")
            anticipation: Number of periods before treatment where effects may appear
            clustervars: Variables for clustering standard errors
            est_method: Estimation method - "dr" (doubly robust), "ipw", or "reg"
            bstrap: Use bootstrap for inference
            biters: Number of bootstrap iterations
            alp: Significance level for confidence intervals
            cband: Compute uniform confidence bands
            weightsname: Variable name for sample weights
            panel: Is data panel (vs repeated cross-sections)
            allow_unbalanced_panel: Allow unbalanced panel data
            base_period: "varying" or "universal" base period
            aggregation_type: "dynamic", "group", "calendar", or "simple"
            
        Returns:
            Dict with CS estimation results based on aggregation type
        """
        if not R_AVAILABLE or 'did' not in self.r_packages:
            return {
                "status": "error",
                "message": "did R package not available"
            }
        
        try:
            logger.info(f"Running enhanced Callaway & Sant'Anna estimator (method={est_method}, aggregation={aggregation_type})")
            
            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)
            
            # Get functions
            att_gt = self.r_packages['did'].att_gt
            aggte = self.r_packages['did'].aggte
            
            # Build att_gt arguments dynamically
            att_gt_args = {
                "yname": yname,
                "tname": tname,
                "idname": idname,
                "gname": gname,
                "data": r_data,
                "control_group": control_group,
                "anticipation": anticipation,
                "est_method": est_method,
                "bstrap": bstrap,
                "biters": biters,
                "alp": alp,
                "cband": cband,
                "panel": panel,
                "allow_unbalanced_panel": allow_unbalanced_panel,
                "base_period": base_period
            }
            
            # Add optional parameters only if provided
            if xformla:
                att_gt_args["xformla"] = robjects.Formula(xformla)
            else:
                att_gt_args["xformla"] = robjects.NULL
                
            if clustervars:
                att_gt_args["clustervars"] = clustervars
                
            if weightsname:
                att_gt_args["weightsname"] = weightsname
            
            # Step 1: Compute group-time ATT with all parameters
            logger.info(f"Computing group-time ATT with anticipation={anticipation}, est_method={est_method}")
            cs_att = att_gt(**att_gt_args)

            # Step 2: ALWAYS compute both "group" and "dynamic" aggregations
            # - "group" gives correct overall ATT (Callaway & Sant'Anna 2021 recommendation)
            # - "dynamic" gives event study for visualization
            logger.info("Computing 'group' aggregation for overall ATT (official recommendation)")
            cs_agg_group = aggte(cs_att, type="group")

            logger.info("Computing 'dynamic' aggregation for event study")
            cs_agg_dynamic = aggte(cs_att, type="dynamic")

            # Extract results
            result = {
                "status": "success",
                "method": f"Callaway & Sant'Anna (2021) - {est_method.upper()}",
                "aggregation_type": "group+dynamic",  # Now we use both
                "control_group": control_group,
                "anticipation": anticipation,
                "est_method": est_method
            }

            # Get overall ATT from "group" aggregation (OFFICIAL RECOMMENDATION)
            logger.info("Extracting overall ATT from 'group' aggregation")
            result["overall_att"] = {
                "estimate": cs_agg_group.rx2('overall.att')[0],
                "se": cs_agg_group.rx2('overall.se')[0],
                "ci_lower": cs_agg_group.rx2('overall.att')[0] - 1.96 * cs_agg_group.rx2('overall.se')[0],
                "ci_upper": cs_agg_group.rx2('overall.att')[0] + 1.96 * cs_agg_group.rx2('overall.se')[0],
                "pvalue": 2 * (1 - robjects.r['pnorm'](
                    abs(cs_agg_group.rx2('overall.att')[0] / cs_agg_group.rx2('overall.se')[0])
                )[0]),
                "source": "group_aggregation"
            }

            # Extract group-specific effects
            group_effects = {}
            groups = cs_agg_group.rx2('egt')  # Groups
            att_g = cs_agg_group.rx2('att.egt')  # ATT by group
            se_g = cs_agg_group.rx2('se.egt')  # SE by group

            for i, g in enumerate(groups):
                group_effects[int(g)] = {
                    "estimate": att_g[i],
                    "se": se_g[i],
                    "ci_lower": att_g[i] - 1.96 * se_g[i],
                    "ci_upper": att_g[i] + 1.96 * se_g[i]
                }
            result["group_effects"] = group_effects

            # Extract event study from "dynamic" aggregation
            logger.info("Extracting event study from 'dynamic' aggregation")
            event_study = {}
            egt = cs_agg_dynamic.rx2('egt')  # Event time
            att_egt = cs_agg_dynamic.rx2('att.egt')  # ATT by event time
            se_egt = cs_agg_dynamic.rx2('se.egt')  # SE by event time

            for i, e in enumerate(egt):
                # Calculate p-value safely
                if se_egt[i] > 0:
                    z_stat = float(abs(att_egt[i] / se_egt[i]))
                    pval = 2 * (1 - robjects.r['pnorm'](z_stat)[0])
                else:
                    pval = 1.0

                event_study[int(e)] = {
                    "estimate": att_egt[i],
                    "se": se_egt[i],
                    "ci_lower": att_egt[i] - 1.96 * se_egt[i],
                    "ci_upper": att_egt[i] + 1.96 * se_egt[i],
                    "pvalue": pval
                }
            result["event_study"] = event_study

            # Extract full variance-covariance matrix for event study
            try:
                inf_func = cs_agg_dynamic.rx2('inf.function')
                if inf_func and len(inf_func) >= 1:
                    inf_matrix = np.array(inf_func[0])  # First element is usually the influence matrix
                    if inf_matrix.ndim == 2 and inf_matrix.shape[1] == len(egt):
                        n_obs = inf_matrix.shape[0]
                        # Compute variance-covariance matrix from influence functions
                        vcov_raw = np.dot(inf_matrix.T, inf_matrix) / n_obs

                        # Scale to match reported standard errors
                        reported_vars = np.array([se**2 for se in se_egt])
                        diag_vcov = np.diag(vcov_raw)

                        # Calculate average scaling factor
                        valid_ratios = diag_vcov[diag_vcov > 1e-12] / reported_vars[diag_vcov > 1e-12]
                        if len(valid_ratios) > 0:
                            scaling_factor = np.mean(valid_ratios)
                            vcov_scaled = vcov_raw / scaling_factor

                            # Store the full covariance matrix
                            result["event_study_vcov"] = {
                                "matrix": vcov_scaled.tolist(),
                                "periods": [int(e) for e in egt],
                                "scaling_factor": float(scaling_factor),
                                "extraction_method": "influence_functions"
                            }

                            logger.info(f"Extracted event study covariance matrix: {vcov_scaled.shape}")
                            logger.info(f"Scaling factor applied: {scaling_factor:.2f}")
                        else:
                            logger.warning("Could not compute scaling factor for covariance matrix")
                    else:
                        logger.warning(f"Unexpected influence matrix shape: {inf_matrix.shape}")
                else:
                    logger.warning("Could not access influence functions for covariance matrix")

            except Exception as e:
                logger.warning(f"Failed to extract event study covariance matrix: {e}")
                # Fallback to diagonal matrix
                vcov_diag = np.diag([se**2 for se in se_egt])
                result["event_study_vcov"] = {
                    "matrix": vcov_diag.tolist(),
                    "periods": [int(e) for e in egt],
                    "scaling_factor": 1.0,
                    "extraction_method": "diagonal_fallback"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Callaway & Sant'Anna estimation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def sun_abraham_estimator(
        self,
        data: pd.DataFrame,
        formula: str,
        cluster_var: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sun & Abraham (2021) interaction-weighted estimator using direct formula.
        
        Uses fixest::feols() with sunab() for heterogeneity-robust
        event study estimation in staggered DID designs.
        
        Args:
            data: Panel data
            formula: R formula string (e.g., "lemp ~ sunab(first.treat, year) | countyreal + year")
            cluster_var: Variable for clustering standard errors (optional)
            
        Returns:
            Dict with Sun & Abraham estimation results including event study
            
        Examples:
            >>> # Basic usage with dots in column names
            >>> formula = "lemp ~ sunab(first.treat, year) | countyreal + year"
            >>> result = estimator.sun_abraham_estimator(data, formula, cluster_var="countyreal")
            
            >>> # With covariates
            >>> formula = "outcome ~ x1 + x2 + sunab(cohort, time) | unit + time" 
            >>> result = estimator.sun_abraham_estimator(data, formula)
        """
        if not R_AVAILABLE or 'fixest' not in self.r_packages:
            return {
                "status": "error",
                "message": "fixest R package not available"
            }
        
        try:
            logger.info("Running Sun & Abraham estimator with fixest sunab()")

            # CRITICAL VALIDATION: Formula must contain sunab()
            if 'sunab(' not in formula.lower():
                return {
                    "status": "error",
                    "message": (
                        "Sun & Abraham estimator REQUIRES sunab() function in formula.\n\n"
                        "❌ INCORRECT (you provided):\n"
                        f"   '{formula}'\n\n"
                        "✅ CORRECT (required format):\n"
                        "   'outcome ~ sunab(cohort_col, time_col) | unit_col + time_col'\n\n"
                        "Example for mpdta dataset:\n"
                        "   'lemp ~ sunab(first.treat, year) | countyreal + year'\n\n"
                        "WHY sunab() is required:\n"
                        "- sunab() applies interaction weighting (IW) to handle heterogeneous treatment effects\n"
                        "- Regular i() interactions do NOT implement the Sun & Abraham (2021) method\n"
                        "- Using i() may produce biased estimates due to negative weights\n\n"
                        "Syntax: sunab(cohort_col, time_col)\n"
                        "  - cohort_col: First treatment period (0 for never-treated)\n"
                        "  - time_col: Time variable\n\n"
                        "For more details, see: ?fixest::sunab in R"
                    ),
                    "provided_formula": formula,
                    "reference": "Sun & Abraham (2021) Journal of Econometrics"
                }

            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)

            # Get fixest functions
            fixest = self.r_packages['fixest']
            feols = fixest.feols

            # Use formula directly without parsing/rebuilding
            formula_str = formula.strip()
            logger.info(f"Sun & Abraham formula (validated): {formula_str}")
            
            # Run Sun & Abraham estimation
            if cluster_var:
                sa_result = feols(
                    robjects.Formula(formula_str),
                    data=r_data,
                    cluster=cluster_var
                )
            else:
                sa_result = feols(
                    robjects.Formula(formula_str),
                    data=r_data
                )
            
            logger.info("Sun & Abraham estimation completed, extracting coefficients...")
            
            # Extract coefficient table using fixest methods
            coef_table = fixest.coeftable(sa_result)
            
            # Get row names before conversion
            try:
                row_names = list(robjects.r['rownames'](coef_table))
            except:
                row_names = None
            
            # Convert to pandas dataframe for easier processing
            with localconverter(robjects.default_converter + pandas2ri.converter):
                coef_df = robjects.conversion.rpy2py(coef_table)
            
            # Handle case where conversion returns numpy array instead of DataFrame
            if not isinstance(coef_df, pd.DataFrame):
                # If it's a numpy array, convert to DataFrame with proper columns
                if hasattr(coef_df, 'shape'):
                    # Assume standard coefficient table format
                    column_names = ['Estimate', 'Std. Error', 't value', 'Pr(>|t|)']
                    coef_df = pd.DataFrame(coef_df, columns=column_names[:coef_df.shape[1]])
                    # Add row names if we got them
                    if row_names:
                        coef_df['coefficient'] = row_names
                    else:
                        # Fallback to indexed names
                        coef_df['coefficient'] = [f'sunab::{i-3}' for i in range(len(coef_df))]
            
            logger.info(f"Coefficient table shape: {coef_df.shape}")
            # Safely get columns
            if hasattr(coef_df, 'columns'):
                logger.info(f"Coefficient columns: {coef_df.columns.tolist()}")
            else:
                logger.info(f"Coefficient type: {type(coef_df)}")
            
            # Extract event study results
            event_study = {}
            overall_estimates = []
            
            for idx, row in coef_df.iterrows():
                coef_name = str(row.get('coefficient', ''))
                estimate = float(row['Estimate'])
                std_error = float(row['Std. Error'])
                pvalue = float(row['Pr(>|t|)'])
                
                # Parse coefficient name to extract relative time period
                # sunab coefficients can have format like "sunab::X", "time::X", or "year::X" (fixest format)
                if 'sunab' in coef_name or 'time' in coef_name or 'year::' in coef_name:
                    try:
                        # Extract relative time period from coefficient name  
                        # Different possible formats: "sunab::X", "time::X", "year::X", "sunab:X", or just numbers
                        if "::" in coef_name:
                            # Format: "sunab::X", "time::X", or "year::X" where X is relative time
                            rel_time_str = coef_name.split("::")[-1]
                        elif ":" in coef_name:
                            # Format: "sunab:X" or "time:X"
                            rel_time_str = coef_name.split(":")[-1]
                        else:
                            # Try to extract number from coefficient name
                            import re
                            match = re.search(r'([+-]?\d+)', coef_name)
                            rel_time_str = match.group(1) if match else "0"
                        
                        rel_time = int(float(rel_time_str))
                        
                        event_study[rel_time] = {
                            "estimate": estimate,
                            "se": std_error,
                            "ci_lower": estimate - 1.96 * std_error,
                            "ci_upper": estimate + 1.96 * std_error,
                            "pvalue": pvalue
                        }
                        
                        # Collect post-treatment estimates for overall ATT
                        if rel_time >= 0:
                            overall_estimates.append(estimate)
                            
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse Sun & Abraham coefficient {coef_name}: {e}")
                        continue
            
            # Calculate overall ATT using fixest aggregation if possible
            try:
                logger.info("Attempting to get overall ATT using fixest summary(agg='ATT')")
                # Try to get aggregated ATT using fixest's built-in aggregation
                att_result = robjects.r('summary')(sa_result, agg="ATT")
                logger.info(f"ATT result obtained, type: {type(att_result)}")

                # Extract ATT from aggregated result
                att_coef_table = fixest.coeftable(att_result)
                logger.info(f"ATT coef_table obtained, type: {type(att_coef_table)}")

                with localconverter(robjects.default_converter + pandas2ri.converter):
                    att_df = robjects.conversion.rpy2py(att_coef_table)

                logger.info(f"Converted to Python, type: {type(att_df)}")

                if isinstance(att_df, pd.DataFrame):
                    logger.info(f"DataFrame shape: {att_df.shape}, columns: {att_df.columns.tolist()}")
                    logger.info(f"DataFrame empty: {att_df.empty}")
                    if not att_df.empty:
                        logger.info(f"DataFrame content:\n{att_df}")
                        overall_att_est = float(att_df.iloc[0]['Estimate'])
                        overall_att_se = float(att_df.iloc[0]['Std. Error'])
                        overall_att_pval = float(att_df.iloc[0]['Pr(>|t|)'])
                        logger.info(f"✅ Successfully extracted ATT: {overall_att_est:.4f} (SE: {overall_att_se:.4f})")
                    else:
                        raise ValueError("DataFrame is empty")
                else:
                    # Not a DataFrame, might be numpy array
                    logger.warning(f"Conversion did not produce DataFrame, got {type(att_df)}")
                    if hasattr(att_df, 'shape') and att_df.shape[0] > 0:
                        logger.info(f"Array shape: {att_df.shape}, attempting to extract values")
                        # Assume first row is ATT with columns: Estimate, Std. Error, t value, Pr(>|t|)
                        overall_att_est = float(att_df[0, 0])  # First column = Estimate
                        overall_att_se = float(att_df[0, 1])   # Second column = Std. Error
                        overall_att_pval = float(att_df[0, 3]) # Fourth column = p-value
                        logger.info(f"✅ Extracted ATT from array: {overall_att_est:.4f} (SE: {overall_att_se:.4f})")
                    else:
                        raise ValueError(f"Cannot extract ATT from type {type(att_df)}")

            except Exception as e:
                logger.error(f"❌ fixest ATT aggregation failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # BUG FIX #1: Remove incorrect fallback aggregation
                # The fallback used np.mean(overall_estimates) which violates
                # Sun & Abraham (2021) cohort-weighted aggregation methodology.
                # Instead, return error to force proper aggregation.
                return {
                    "status": "error",
                    "message": f"Failed to extract overall ATT from fixest summary(agg='ATT'): {str(e)}. "
                               f"Event study estimates are available but overall ATT aggregation failed. "
                               f"This may indicate an issue with the fixest package version or formula specification."
                }
            
            overall_att = {
                "estimate": float(overall_att_est),
                "se": float(overall_att_se),
                "ci_lower": float(overall_att_est - 1.96 * overall_att_se),
                "ci_upper": float(overall_att_est + 1.96 * overall_att_se),
                "pvalue": float(overall_att_pval)
            }
            
            # Get number of observations
            try:
                n_obs = int(sa_result.rx2('nobs')[0])
            except:
                n_obs = len(data)
            
            logger.info(f"Sun & Abraham estimation completed with {len(event_study)} time periods")
            
            return {
                "status": "success",
                "method": "Sun & Abraham (2021) Interaction-Weighted",
                "overall_att": overall_att,
                "event_study": event_study,
                "n_periods": len(event_study),
                "formula": formula_str,
                "n_obs": n_obs
            }
            
        except Exception as e:
            logger.error(f"Error in Sun & Abraham estimation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def bjs_imputation_estimator(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        cohort_col: str,  # REQUIRED - not Optional anymore
        horizon: int = 10,
        pretrends_test: bool = True
    ) -> Dict[str, Any]:
        """
        Borusyak, Jaravel & Spiess (2024) imputation estimator.

        Uses didimputation package for efficient computation with
        staggered treatment timing. Implements event study estimation
        using imputation-based approach.

        IMPORTANT: BJS requires a COHORT variable (first treatment time),
        NOT a binary 0/1 treatment indicator!

        Args:
            data: Panel data
            outcome_col: Outcome variable name
            unit_col: Unit identifier
            time_col: Time variable
            cohort_col: Cohort/group variable indicating FIRST TREATMENT TIME
                       (e.g., 'first.treat' with values 0, 2004, 2006, 2007)
                       Value 0 indicates never-treated units
            horizon: Maximum horizon for event study or True for full event study
            pretrends_test: Whether to test for pre-trends

        Returns:
            Dict with BJS imputation results including event study estimates
        """
        if not R_AVAILABLE or 'didimputation' not in self.r_packages:
            return {
                "status": "error",
                "message": "didimputation R package not available"
            }

        try:
            logger.info("Running BJS imputation estimator with didimputation package")

            # VALIDATE: cohort_col must be provided and valid
            if cohort_col is None or cohort_col not in data.columns:
                return {
                    "status": "error",
                    "message": (
                        "BJS imputation requires 'cohort_col' parameter with COHORT variable (first treatment time).\n\n"
                        "❌ INCORRECT: Using binary treatment indicator (0/1)\n"
                        "✅ CORRECT: Using cohort variable like 'first.treat' with values (0, 2004, 2006, 2007)\n\n"
                        "Example: cohort_col='first.treat'\n"
                        "  - 0 = never treated\n"
                        "  - 2004, 2006, 2007 = year first treated\n\n"
                        "This is a fundamental requirement of the BJS (2024) method."
                    ),
                    "reference": "Borusyak, Jaravel & Spiess (2024)"
                }

            # VALIDATE: cohort variable has proper structure (not just 0/1)
            unique_cohorts = data[cohort_col].nunique()
            unique_values = sorted(data[cohort_col].unique())

            if unique_cohorts == 2 and set(unique_values) == {0, 1}:
                return {
                    "status": "error",
                    "message": (
                        f"cohort_col '{cohort_col}' appears to be a binary treatment indicator (0/1), "
                        "not a cohort variable.\n\n"
                        "BJS imputation requires a COHORT VARIABLE indicating WHEN treatment first occurred.\n\n"
                        "❌ Binary indicator (0/1) - NOT valid\n"
                        "✅ Cohort variable (0, 2004, 2006, 2007) - Valid\n\n"
                        "Please provide the correct cohort column (e.g., 'first.treat', 'cohort', 'gvar')."
                    )
                }

            if unique_cohorts <= 2:
                logger.warning(
                    f"cohort_col '{cohort_col}' has only {unique_cohorts} unique values: {unique_values}. "
                    "BJS works best with multiple cohorts. Proceeding but results may be limited."
                )

            gname_var = cohort_col
            logger.info(f"Using cohort variable: {gname_var} with {unique_cohorts} cohorts: {unique_values}")
            
            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)
            
            # Get did_imputation function
            did_imputation = self.r_packages['didimputation'].did_imputation

            # STEP 1: Run STATIC mode for overall ATT (OFFICIAL RECOMMENDATION)
            logger.info("Running did_imputation in STATIC mode for overall ATT (official recommendation)")
            result_static = did_imputation(
                data=r_data,
                yname=outcome_col,
                idname=unit_col,
                tname=time_col,
                gname=gname_var
                # horizon not specified = Static mode
            )

            # STEP 2: Run EVENT STUDY mode for visualization
            logger.info("Running did_imputation in EVENT STUDY mode with pre-trends")
            result_event = did_imputation(
                data=r_data,
                yname=outcome_col,
                idname=unit_col,
                tname=time_col,
                gname=gname_var,
                horizon=True,  # Enable full event study analysis
                pretrends=True  # Include pre-treatment estimates for HonestDiD compatibility
            )

            logger.info("BJS estimation completed, extracting results...")

            # Convert STATIC result for overall ATT
            with localconverter(robjects.default_converter + pandas2ri.converter):
                result_static_df = robjects.conversion.rpy2py(result_static)

            logger.info(f"Static result shape: {result_static_df.shape}")
            logger.info(f"Static result columns: {result_static_df.columns.tolist()}")

            # Convert EVENT STUDY result for visualization
            with localconverter(robjects.default_converter + pandas2ri.converter):
                result_event_df = robjects.conversion.rpy2py(result_event)

            logger.info(f"Event study result shape: {result_event_df.shape}")
            logger.info(f"Event study result columns: {result_event_df.columns.tolist()}")

            # CRITICAL CHECK: Validate results are not empty
            if len(result_static_df) == 0:
                logger.error("BJS estimation returned empty result")
                return {
                    "status": "error",
                    "message": (
                        "BJS estimation returned empty result. This typically occurs when:\n\n"
                        "1. cohort_col is a binary treatment indicator (0/1) instead of cohort variable\n"
                        "2. All units are treated at the same time (no cohort variation)\n"
                        "3. Data structure is incompatible with BJS assumptions\n\n"
                        "Solution: Use a proper cohort variable indicating FIRST TREATMENT TIME.\n"
                        f"Current cohort_col: '{gname_var}'\n"
                        f"Unique values: {sorted(data[gname_var].unique())}\n\n"
                        "Expected format: 0 (never-treated) and treatment years (e.g., 2004, 2006, 2007)"
                    ),
                    "cohort_values": sorted(data[gname_var].unique().tolist())
                }

            if len(result_event_df) == 0:
                logger.warning("BJS event study returned empty result")

            # EXTRACT OVERALL ATT from STATIC mode (OFFICIAL RECOMMENDATION)
            static_att = float(result_static_df['estimate'].iloc[0])
            static_se = float(result_static_df['std.error'].iloc[0])
            static_ci_low = float(result_static_df['conf.low'].iloc[0])
            static_ci_high = float(result_static_df['conf.high'].iloc[0])

            logger.info(f"BJS Static Mode - Overall ATT: {static_att:.6f} (SE: {static_se:.6f})")

            # EXTRACT EVENT STUDY results from EVENT STUDY mode (for visualization)
            event_study = {}
            overall_estimates = []
            nan_count = 0

            for idx, row in result_event_df.iterrows():
                term = int(row['term'])
                estimate = float(row['estimate'])
                std_error = float(row['std.error'])
                conf_low = float(row['conf.low'])
                conf_high = float(row['conf.high'])

                # Track NaN values
                if np.isnan(estimate):
                    nan_count += 1
                    logger.warning(f"BJS Event Study: Period {term} has NaN estimate")

                # Calculate p-value safely (handle NaN)
                if not np.isnan(std_error) and std_error > 0 and not np.isnan(estimate):
                    z_stat = float(abs(estimate / std_error))
                    pval = float(2 * (1 - robjects.r['pnorm'](z_stat)[0]))
                else:
                    pval = 1.0

                event_study[term] = {
                    "estimate": estimate,
                    "se": std_error,
                    "ci_lower": conf_low,
                    "ci_upper": conf_high,
                    "pvalue": pval
                }

                # Collect post-treatment estimates for overall ATT (including NaN for now)
                if term >= 0:
                    overall_estimates.append(estimate)

            # Log NaN summary for event study
            if nan_count > 0:
                logger.warning(f"BJS Event Study: {nan_count}/{len(result_event_df)} periods have NaN estimates")

            # USE STATIC MODE RESULT for overall ATT (OFFICIAL RECOMMENDATION)
            # No need for complex weighted averaging - Static mode returns it directly!
            logger.info("Using Static Mode result for overall ATT (official BJS 2024 recommendation)")

            # Calculate p-value from Static mode result
            if static_se > 0 and not np.isnan(static_att) and not np.isnan(static_se):
                z_stat = float(abs(static_att / static_se))
                overall_pval = float(2 * (1 - robjects.r['pnorm'](z_stat)[0]))
            else:
                overall_pval = 1.0

            overall_att = {
                "estimate": static_att,
                "se": static_se,
                "ci_lower": static_ci_low,
                "ci_upper": static_ci_high,
                "pvalue": overall_pval,
                "source": "static_mode"  # Indicate this is from Static mode
            }
            
            # Pre-trends test - test if pre-treatment effects are jointly zero
            pretrends_result = None
            if pretrends_test:
                pre_treatment_effects = [event_study[t]["estimate"] for t in event_study.keys() if t < 0]
                if len(pre_treatment_effects) > 0:
                    # Simple test: check if any pre-treatment effect is statistically significant
                    pre_treatment_pvals = [event_study[t]["pvalue"] for t in event_study.keys() if t < 0]
                    min_pval = min(pre_treatment_pvals) if pre_treatment_pvals else 1.0
                    pretrends_result = {
                        "pvalue": float(min_pval),
                        "passed": min_pval > 0.05,
                        "test_method": "minimum_pvalue_pretreatment"
                    }
                    
            logger.info(f"BJS estimation completed successfully with {len(event_study)} periods")
            
            return {
                "status": "success",
                "method": "Borusyak, Jaravel & Spiess (2024)",
                "overall_att": overall_att,
                "event_study": event_study,
                "pretrends_test": pretrends_result,
                "n_periods": len(event_study),
                "gname_variable": gname_var
            }
            
        except Exception as e:
            logger.error(f"Error in BJS imputation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def gardner_two_stage_estimator(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        cohort_col: str,  # REQUIRED - not Optional anymore
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Gardner (2022) two-stage DID estimator with EVENT STUDY specification.

        Uses did2s package for two-stage estimation that is robust to
        heterogeneous treatment effects. ALWAYS uses event study specification
        because static treatment effects are not identified with unit+time FE.

        IMPORTANT: Gardner's method REQUIRES cohort variable for event study.
        Static treatment indicators cause collinearity with fixed effects.

        Args:
            data: Panel data
            outcome_col: Outcome variable
            unit_col: Unit identifier
            time_col: Time variable
            cohort_col: Cohort/group variable indicating FIRST TREATMENT TIME
                       (e.g., 'first.treat' with values 0, 2004, 2006, 2007)
                       REQUIRED for proper identification
            covariates: Optional list of covariates

        Returns:
            Dict with Gardner two-stage EVENT STUDY results
        """
        if not R_AVAILABLE or 'did2s' not in self.r_packages:
            return {
                "status": "error",
                "message": "did2s R package not available"
            }

        try:
            logger.info("Running Gardner two-stage estimator with did2s package (EVENT STUDY mode)")

            # VALIDATE: cohort_col is REQUIRED for identification
            if cohort_col is None or cohort_col not in data.columns:
                return {
                    "status": "error",
                    "message": (
                        "Gardner two-stage estimator REQUIRES 'cohort_col' for proper identification.\n\n"
                        "❌ PROBLEM: Static treatment indicators are COLLINEAR with unit + time fixed effects.\n"
                        "   After removing FE in stage 1, no variation remains in the treatment variable.\n\n"
                        "✅ SOLUTION: Use event study specification with cohort variable.\n\n"
                        "Example: cohort_col='first.treat' (values: 0, 2004, 2006, 2007)\n"
                        "  - 0 = never treated\n"
                        "  - 2004, 2006, 2007 = year first treated\n\n"
                        "Gardner (2022) is designed for EVENT STUDY analysis, not static ATT.\n"
                        "For simple ATT, use Callaway & Sant'Anna or Efficient estimator instead."
                    ),
                    "recommendation": "callaway_santanna",
                    "reference": "Gardner (2022) 'Two-stage differences in differences'"
                }

            # Create data with relative time variables for event study
            data_copy = data.copy()

            # Create relative time variable
            if cohort_col in data.columns:
                # Create relative time: current time - treatment time
                data_copy['rel_time'] = data_copy[time_col] - data_copy[cohort_col]
                # Set never-treated units to have rel_time = Inf (official did2s convention)
                # CRITICAL: Must use Inf (not -999) to match official did2s implementation
                # Inf is excluded from regression via ref = c(-1, Inf) in second_stage
                never_treated_mask = (data_copy[cohort_col].isna()) | (data_copy[cohort_col] == 0) | (data_copy[cohort_col] >= 9999)
                data_copy.loc[never_treated_mask, 'rel_time'] = np.inf

                # CRITICAL: Create correct treatment indicator (year >= first_treat)
                data_copy['_did2s_treat'] = 0  # Default to untreated
                treated_mask = (data_copy[cohort_col] > 0) & (data_copy[cohort_col] < 9999) & (data_copy[time_col] >= data_copy[cohort_col])
                data_copy.loc[treated_mask, '_did2s_treat'] = 1
                treatment_var = '_did2s_treat'

                logger.info("Using event study approach with relative time variable")
                logger.info(f"Created correct treatment indicator: {treatment_var}")
                logger.info(f"Relative time range: {data_copy['rel_time'].min():.0f} to {data_copy['rel_time'].max():.0f}")
            
            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data_copy)
            
            # Get did2s function
            did2s = self.r_packages['did2s'].did2s
            
            # Build formulas for two-stage estimation
            # First stage: fixed effects and covariates
            if covariates:
                first_stage_vars = [unit_col, time_col] + covariates
                first_stage_formula = f"~ 0 | {' + '.join(first_stage_vars)}"
            else:
                first_stage_formula = f"~ 0 | {unit_col} + {time_col}"

            # Second stage: EVENT STUDY specification (always)
            # Use i() function for relative time with reference periods at -1 AND Inf
            # CRITICAL: Must exclude BOTH -1 (pre-treatment) and Inf (never-treated)
            # to match official did2s implementation and avoid biased coefficients
            second_stage_formula = "~ i(rel_time, ref = c(-1, Inf))"

            logger.info(f"First stage formula: {first_stage_formula}")
            logger.info(f"Second stage formula: {second_stage_formula}")

            # Calculate sample sizes for each rel_time period (for weighted ATT)
            # Weights should be N_gt / N_post as per Gardner (2022)
            rel_time_counts = {}
            for rel_time_val in data_copy['rel_time'].unique():
                if not np.isinf(rel_time_val) and rel_time_val >= 0:  # Only post-treatment periods
                    count = ((data_copy['rel_time'] == rel_time_val) &
                            (data_copy[treatment_var] == 1)).sum()
                    rel_time_counts[int(rel_time_val)] = int(count)
            logger.info(f"Post-treatment sample sizes by period: {rel_time_counts}")

            # Run Gardner two-stage estimation
            result = did2s(
                data=r_data,
                yname=outcome_col,
                first_stage=robjects.Formula(first_stage_formula),
                second_stage=robjects.Formula(second_stage_formula),
                treatment=treatment_var,  # Use the correctly defined treatment variable
                cluster_var=unit_col
            )
            
            logger.info("Gardner estimation completed, extracting results...")
            
            # Extract results from fixest object using standard methods
            # Get coefficients table
            fixest_pkg = self.r_packages.get('fixest', robjects.packages.importr('fixest'))
            
            # Extract coefficients and standard errors
            coef_table = fixest_pkg.coeftable(result)
            
            # Convert coefficient table to easier format
            coef_names = list(coef_table.rownames)
            estimates = list(coef_table.rx(True, 1))  # Estimates column
            std_errors = list(coef_table.rx(True, 2))  # Std.Error column
            pvalues = list(coef_table.rx(True, 4))     # Pr(>|t|) column
            
            logger.info(f"Extracted {len(coef_names)} coefficients")
            logger.info(f"Coefficient names: {coef_names}")
            logger.info("Using EVENT STUDY specification (always)")

            # Organize results
            event_study = {}
            overall_estimates = []
            overall_estimates_weighted = []  # For weighted average using N_gt/N_post
            overall_att_est = 0.0
            att_se = 0.0

            for i, coef_name in enumerate(coef_names):
                estimate = float(estimates[i])
                se = float(std_errors[i])
                pval = float(pvalues[i])

                logger.info(f"Processing coefficient [{i}]: '{coef_name}' = {estimate} (SE: {se})")

                # Parse coefficient name to extract time period (EVENT STUDY always)
                if 'rel_time' in coef_name:
                    # Extract relative time from coefficient name (e.g., "rel_time::0" -> 0)
                    try:
                        if "::" in coef_name:
                            rel_time = int(float(coef_name.split("::")[-1]))
                        else:
                            # Parse other potential formats
                            import re
                            match = re.search(r'rel_time.*?([+-]?\d+)', coef_name)
                            if match:
                                rel_time = int(match.group(1))
                            else:
                                rel_time = 0

                        # SAFETY CHECK: Inf should not appear since it's excluded in ref = c(-1, Inf)
                        # This check remains as a safeguard in case the R formula doesn't work as expected
                        if np.isinf(rel_time):
                            logger.warning(f"Unexpected: rel_time = Inf found in coefficients (should be excluded by ref parameter)")
                            continue  # Skip this coefficient

                        event_study[rel_time] = {
                            "estimate": estimate,
                            "se": se,
                            "ci_lower": estimate - 1.96 * se,
                            "ci_upper": estimate + 1.96 * se,
                            "pvalue": pval
                        }

                        # Collect post-treatment estimates for overall ATT (weighted by N_gt)
                        if rel_time >= 0:
                            overall_estimates.append(estimate)
                            # Add weighted estimate (estimate * sample_size for this period)
                            if rel_time in rel_time_counts:
                                overall_estimates_weighted.append((estimate, rel_time_counts[rel_time]))

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse coefficient {coef_name}: {e}")

            # Calculate overall ATT for event study (weighted by sample sizes)
            if overall_estimates_weighted:
                # Use weighted average: ATT = Σ(ATT_t * N_t) / Σ(N_t)
                # This follows Gardner (2022): weights = N_gt / N_post
                total_weight = sum(weight for _, weight in overall_estimates_weighted)
                overall_att_est = sum(est * weight for est, weight in overall_estimates_weighted) / total_weight

                # Use SE from period 0 or first available post-treatment period
                post_treatment_effects = {k: v for k, v in event_study.items() if k >= 0}
                if post_treatment_effects:
                    att_se = list(post_treatment_effects.values())[0]["se"]
                else:
                    att_se = np.mean([v["se"] for v in event_study.values()])

                simple_mean = np.mean(overall_estimates) if overall_estimates else 0.0
                logger.info(f"Event study: ATT (weighted by N_gt/N_post) = {overall_att_est:.4f}")
                logger.info(f"Event study: ATT (simple mean, for comparison) = {simple_mean:.4f}")
                logger.info(f"Total post-treatment observations: {total_weight}")
            # For static model, overall_att_est and att_se already set in loop above
                
            # Calculate p-value safely avoiding numpy.float64 issues
            if att_se > 0:
                z_stat = float(abs(overall_att_est / att_se))
                pval = float(2 * (1 - robjects.r['pnorm'](z_stat)[0]))
            else:
                pval = 1.0
                
            overall_att = {
                "estimate": float(overall_att_est),
                "se": float(att_se),
                "ci_lower": float(overall_att_est - 1.96 * att_se),
                "ci_upper": float(overall_att_est + 1.96 * att_se),
                "pvalue": pval
            }
            
            logger.info(f"Gardner estimation completed with {len(event_study)} time periods")
            
            return {
                "status": "success",
                "method": "Gardner (2022) Two-Stage DiD (Event Study)",
                "overall_att": overall_att,
                "event_study": event_study,
                "n_periods": len(event_study),
                "specification": "event_study"
            }
            
        except Exception as e:
            logger.error(f"Error in Gardner two-stage estimation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def honest_did_sensitivity_analysis(
        self,
        betahat: np.ndarray,
        sigma: np.ndarray,
        num_pre_periods: int,
        num_post_periods: int,
        method: str = "relative_magnitude",
        m_values: Optional[List[float]] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        HonestDiD sensitivity analysis implementation.
        
        Args:
            betahat: Event study coefficients
            sigma: Variance-covariance matrix
            num_pre_periods: Number of pre-treatment periods
            num_post_periods: Number of post-treatment periods  
            method: "relative_magnitude" or "smoothness"
            m_values: List of M values for sensitivity analysis
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict with sensitivity analysis results
        """
        if not R_AVAILABLE or 'HonestDiD' not in self.r_packages:
            return {
                "status": "error",
                "message": "HonestDiD R package not available"
            }
        
        try:
            logger.info(f"Running HonestDiD sensitivity analysis with {method}")
            logger.info(f"Input dimensions: betahat={betahat.shape}, sigma={sigma.shape}")
            logger.info(f"Periods: {num_pre_periods} pre, {num_post_periods} post")
            
            # Validate inputs
            if len(betahat) != num_pre_periods + num_post_periods:
                return {
                    "status": "error",
                    "message": f"betahat length ({len(betahat)}) doesn't match periods ({num_pre_periods + num_post_periods})"
                }
            
            if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != len(betahat):
                return {
                    "status": "error", 
                    "message": f"sigma matrix shape {sigma.shape} incompatible with betahat length {len(betahat)}"
                }
            
            # Set default M values based on best practices
            if m_values is None:
                m_values = [0.5, 1.0, 1.5, 2.0]  # Standard setting from research
            
            # Get HonestDiD functions
            honest_did = self.r_packages['HonestDiD']
            
            # Convert inputs to R with explicit formatting
            r_betahat = robjects.FloatVector(betahat.flatten())
            r_sigma = robjects.r['matrix'](
                robjects.FloatVector(sigma.flatten()),
                nrow=sigma.shape[0],
                ncol=sigma.shape[1]
            )
            
            alpha = 1 - confidence_level
            
            logger.info("Creating original confidence set...")
            
            # Debug: Check if HonestDiD functions exist
            try:
                # Test function availability
                r_func_test = robjects.r('exists("constructOriginalCS", where="package:HonestDiD")')
                logger.info(f"constructOriginalCS exists: {r_func_test[0]}")
            except Exception as e:
                logger.warning(f"Could not check function existence: {e}")
            
            # Create original confidence set with error checking
            try:
                original_cs = honest_did.constructOriginalCS(
                    betahat=r_betahat,
                    sigma=r_sigma,
                    numPrePeriods=num_pre_periods,
                    numPostPeriods=num_post_periods,
                    alpha=alpha
                )
                
                # Check if result is NULL
                if str(type(original_cs)).find('NULLType') >= 0:
                    return {
                        "status": "error",
                        "message": "constructOriginalCS returned NULL - check HonestDiD package installation and parameters"
                    }
                    
                logger.info(f"Original CS created successfully: {type(original_cs)}")
                
            except Exception as e:
                logger.error(f"constructOriginalCS failed: {e}")
                return {
                    "status": "error",
                    "message": f"constructOriginalCS failed: {str(e)}"
                }
            
            if method == "relative_magnitude":
                # Relative magnitude constraints
                r_m_values = robjects.FloatVector(m_values)
                
                logger.info(f"Running sensitivity analysis for M values: {m_values}")
                
                # Run sensitivity analysis with error checking
                try:
                    sensitivity_results_r = honest_did.createSensitivityResults_relativeMagnitudes(
                        betahat=r_betahat,
                        sigma=r_sigma,
                        numPrePeriods=num_pre_periods,
                        numPostPeriods=num_post_periods,
                        Mbarvec=r_m_values,
                        alpha=alpha
                    )
                    
                    # Check if result is NULL
                    if str(type(sensitivity_results_r)).find('NULLType') >= 0:
                        return {
                            "status": "error",
                            "message": "createSensitivityResults_relativeMagnitudes returned NULL - check parameters and data"
                        }
                        
                    logger.info(f"Sensitivity results created: {type(sensitivity_results_r)}")
                    
                except Exception as e:
                    logger.error(f"createSensitivityResults_relativeMagnitudes failed: {e}")
                    return {
                        "status": "error",
                        "message": f"Sensitivity analysis failed: {str(e)}"
                    }
                
                # Extract breakdown point and intervals with improved error handling
                breakdown_point = None
                robust_intervals = {}
                
                try:
                    # Check if sensitivity_results_r has the expected structure
                    logger.info("Extracting sensitivity results...")
                    
                    # Try to get the names/structure of the result
                    try:
                        result_names = robjects.r('names')(sensitivity_results_r)
                        logger.info(f"Result structure names: {list(result_names)}")
                    except:
                        logger.warning("Could not extract result names")
                    
                    # Extract confidence intervals - HonestDiD uses 'lb' and 'ub' 
                    lower_ci = sensitivity_results_r.rx2('lb')
                    upper_ci = sensitivity_results_r.rx2('ub')
                    
                    if lower_ci is None or upper_ci is None:
                        # Try alternative names
                        try:
                            lower_ci = sensitivity_results_r.rx2('lowerCI')
                            upper_ci = sensitivity_results_r.rx2('upperCI')
                        except:
                            return {
                                "status": "error",
                                "message": "Could not extract confidence intervals from sensitivity results"
                            }
                    
                    for i, m in enumerate(m_values):
                        try:
                            ci_lower = float(lower_ci[i])
                            ci_upper = float(upper_ci[i])
                            contains_zero = ci_lower <= 0 <= ci_upper
                            
                            robust_intervals[f"M_{m}"] = {
                                "m_value": m,
                                "ci_lower": ci_lower,
                                "ci_upper": ci_upper,
                                "contains_zero": contains_zero
                            }
                            
                            # First M where result becomes insignificant
                            if breakdown_point is None and contains_zero:
                                breakdown_point = m
                                
                        except Exception as e:
                            logger.warning(f"Could not extract results for M={m}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error extracting sensitivity intervals: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to extract sensitivity results: {str(e)}"
                    }
                
                # Extract original CI with error handling
                try:
                    # HonestDiD uses 'lb' and 'ub' for original confidence sets too
                    original_ci = {
                        "ci_lower": float(original_cs.rx2('lb')[0]),
                        "ci_upper": float(original_cs.rx2('ub')[0])
                    }
                except Exception as e:
                    try:
                        # Try alternative names
                        original_ci = {
                            "ci_lower": float(original_cs.rx2('lowerCI')[0]),
                            "ci_upper": float(original_cs.rx2('upperCI')[0])
                        }
                    except Exception as e2:
                        logger.warning(f"Could not extract original CI: {e}, {e2}")
                        original_ci = {"ci_lower": None, "ci_upper": None}
                
                return {
                    "status": "success",
                    "method": "relative_magnitude",
                    "breakdown_point": breakdown_point,
                    "robust_intervals": robust_intervals,
                    "original_ci": original_ci,
                    "m_values": m_values,
                    "num_pre_periods": num_pre_periods,
                    "num_post_periods": num_post_periods
                }
                
            elif method == "smoothness":
                # Smoothness constraints - restrict changes in slope between periods
                # Default M values for smoothness (smaller values since measuring slope changes)
                if m_values is None:
                    m_values = [0.01, 0.02, 0.03, 0.04, 0.05]  # Standard smoothness values
                
                r_m_values = robjects.FloatVector(m_values)
                
                logger.info(f"Running smoothness sensitivity analysis for M values: {m_values}")
                
                # Run smoothness sensitivity analysis
                try:
                    # Use createSensitivityResults with smoothness restriction
                    # This restricts how much the slope can change between consecutive periods
                    sensitivity_results_r = honest_did.createSensitivityResults(
                        betahat=r_betahat,
                        sigma=r_sigma,
                        numPrePeriods=num_pre_periods,
                        numPostPeriods=num_post_periods,
                        Mvec=r_m_values,
                        alpha=alpha
                    )
                    
                    # Check if result is NULL
                    if str(type(sensitivity_results_r)).find('NULLType') >= 0:
                        return {
                            "status": "error",
                            "message": "createSensitivityResults (smoothness) returned NULL - check parameters"
                        }
                        
                    logger.info(f"Smoothness sensitivity results created: {type(sensitivity_results_r)}")
                    
                except Exception as e:
                    logger.error(f"createSensitivityResults (smoothness) failed: {e}")
                    return {
                        "status": "error",
                        "message": f"Smoothness sensitivity analysis failed: {str(e)}"
                    }
                
                # Extract breakdown point and intervals
                breakdown_point = None
                robust_intervals = {}
                
                try:
                    logger.info("Extracting smoothness sensitivity results...")
                    
                    # Extract confidence intervals for each M value
                    lower_ci = sensitivity_results_r.rx2('lb')
                    upper_ci = sensitivity_results_r.rx2('ub')
                    
                    if lower_ci is None or upper_ci is None:
                        # Try alternative names
                        try:
                            lower_ci = sensitivity_results_r.rx2('lowerCI')
                            upper_ci = sensitivity_results_r.rx2('upperCI')
                        except:
                            return {
                                "status": "error",
                                "message": "Could not extract confidence intervals from smoothness results"
                            }
                    
                    for i, m in enumerate(m_values):
                        try:
                            ci_lower = float(lower_ci[i])
                            ci_upper = float(upper_ci[i])
                            contains_zero = ci_lower <= 0 <= ci_upper
                            
                            robust_intervals[f"M_{m}"] = {
                                "m_value": m,
                                "ci_lower": ci_lower,
                                "ci_upper": ci_upper,
                                "contains_zero": contains_zero,
                                "interpretation": f"Max slope change: {m} per period"
                            }
                            
                            # First M where result becomes insignificant (breakdown point)
                            if breakdown_point is None and contains_zero:
                                breakdown_point = m
                                
                        except Exception as e:
                            logger.warning(f"Could not extract smoothness results for M={m}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error extracting smoothness intervals: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to extract smoothness results: {str(e)}"
                    }
                
                # Extract original CI
                try:
                    original_ci = {
                        "ci_lower": float(original_cs.rx2('lb')[0]),
                        "ci_upper": float(original_cs.rx2('ub')[0])
                    }
                except Exception as e:
                    try:
                        original_ci = {
                            "ci_lower": float(original_cs.rx2('lowerCI')[0]),
                            "ci_upper": float(original_cs.rx2('upperCI')[0])
                        }
                    except Exception as e2:
                        logger.warning(f"Could not extract original CI: {e}, {e2}")
                        original_ci = {"ci_lower": None, "ci_upper": None}
                
                return {
                    "status": "success",
                    "method": "smoothness",
                    "breakdown_point": breakdown_point,
                    "robust_intervals": robust_intervals,
                    "original_ci": original_ci,
                    "m_values": m_values,
                    "num_pre_periods": num_pre_periods,
                    "num_post_periods": num_post_periods,
                    "interpretation": "M represents maximum change in slope between consecutive periods"
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown sensitivity analysis method: {method}"
                }
                
        except Exception as e:
            logger.error(f"Error in HonestDiD sensitivity analysis: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def pretrends_power_analysis(
        self,
        betahat: np.ndarray,
        sigma: np.ndarray,
        time_vec: np.ndarray,
        reference_period: int = -1,
        target_power: float = 0.8,
        alpha: float = 0.05,
        analyze_slope: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Pretrends power analysis implementation using R pretrends package.
        
        Based on Roth (2022) "Pretest with Caution: Event-Study Estimates 
        after Testing for Parallel Trends"
        
        Args:
            betahat: Event study coefficients
            sigma: Variance-covariance matrix
            time_vec: Vector of relative time periods
            reference_period: Reference period (typically -1)
            target_power: Target power level (0.8 = 80% power)
            alpha: Significance level for hypothesis testing
            analyze_slope: Optional specific slope to analyze (for ex post analysis)
            
        Returns:
            Dict with power analysis results including:
            - slope_for_power: Minimal detectable slope at target power
            - power_statistics: Power metrics for specified slope (if provided)
            - interpretation: Human-readable interpretation
        """
        if not R_AVAILABLE or 'pretrends' not in self.r_packages:
            return {
                "status": "error",
                "message": "pretrends R package not available"
            }
        
        try:
            logger.info(f"Running pretrends power analysis with target_power={target_power}")
            logger.info(f"Input dimensions: betahat={betahat.shape}, sigma={sigma.shape}")
            logger.info(f"Time periods: {time_vec}")
            
            # Validate inputs
            if len(betahat) != len(time_vec):
                return {
                    "status": "error",
                    "message": f"betahat length ({len(betahat)}) doesn't match time_vec length ({len(time_vec)})"
                }
            
            if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != len(betahat):
                return {
                    "status": "error",
                    "message": f"sigma matrix shape {sigma.shape} incompatible with betahat length {len(betahat)}"
                }
            
            # Get pretrends functions
            pretrends_pkg = self.r_packages['pretrends']
            
            # Convert inputs to R
            r_betahat = robjects.FloatVector(betahat)
            r_sigma = robjects.r['matrix'](
                robjects.FloatVector(sigma.flatten()),
                nrow=sigma.shape[0],
                ncol=sigma.shape[1]
            )
            r_time_vec = robjects.FloatVector(time_vec)
            
            logger.info("Computing slope for target power...")
            
            # Step 1: Calculate slope that would be detected with target power
            try:
                slope_result = pretrends_pkg.slope_for_power(
                    sigma=r_sigma,
                    targetPower=target_power,
                    tVec=r_time_vec,
                    referencePeriod=reference_period
                )
                
                # Extract slope value (always positive in pretrends package)
                minimal_slope = float(slope_result[0])
                logger.info(f"Minimal detectable slope at {target_power*100}% power: {minimal_slope}")
                
            except Exception as e:
                logger.error(f"slope_for_power failed: {e}")
                return {
                    "status": "error",
                    "message": f"Power calculation failed: {str(e)}"
                }
            
            # Step 2: If analyze_slope provided, conduct ex post power analysis
            power_analysis_results = None
            if analyze_slope is not None:
                logger.info(f"Conducting ex post power analysis for slope={analyze_slope}")
                
                try:
                    # Create hypothesized linear trend: slope * time
                    # Exclude reference period from trend
                    hypothesized_trend = []
                    for t in time_vec:
                        if t == reference_period:
                            hypothesized_trend.append(0.0)  # Reference period has no trend
                        else:
                            hypothesized_trend.append(analyze_slope * t)
                    
                    r_hypothesized_trend = robjects.FloatVector(hypothesized_trend)
                    
                    # Run pretrends analysis
                    pretrends_result = pretrends_pkg.pretrends(
                        betahat=r_betahat,
                        sigma=r_sigma,
                        deltatrue=r_hypothesized_trend,
                        tVec=r_time_vec,
                        referencePeriod=reference_period
                    )
                    
                    # Extract power statistics
                    df_power = pretrends_result.rx2('df_power')
                    
                    # Convert R dataframe to Python dict
                    power_metrics = {}
                    if df_power is not None:
                        try:
                            # Extract key power statistics
                            power_prob = float(df_power.rx2('power')[0]) if df_power.rx2('power') else None
                            bayes_factor = float(df_power.rx2('bayes_factor')[0]) if df_power.rx2('bayes_factor') else None
                            likelihood_ratio = float(df_power.rx2('likelihood_ratio')[0]) if df_power.rx2('likelihood_ratio') else None
                            
                            power_metrics = {
                                "power_probability": power_prob,
                                "bayes_factor": bayes_factor,
                                "likelihood_ratio": likelihood_ratio,
                                "analyzed_slope": analyze_slope
                            }
                            
                        except Exception as e:
                            logger.warning(f"Could not extract all power statistics: {e}")
                            power_metrics = {"analyzed_slope": analyze_slope}
                    
                    power_analysis_results = power_metrics
                    
                except Exception as e:
                    logger.error(f"Ex post power analysis failed: {e}")
                    power_analysis_results = {"error": str(e)}
            
            # Step 3: Generate interpretation
            interpretation = self._interpret_pretrends_results(
                minimal_slope, target_power, power_analysis_results
            )
            
            return {
                "status": "success",
                "method": "pretrends_power_analysis",
                "minimal_detectable_slope": minimal_slope,
                "target_power": target_power,
                "reference_period": reference_period,
                "power_analysis": power_analysis_results,
                "interpretation": interpretation,
                "time_periods": time_vec.tolist(),
                "num_periods": len(time_vec)
            }
            
        except Exception as e:
            logger.error(f"Error in pretrends power analysis: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _interpret_pretrends_results(
        self,
        minimal_slope: float,
        target_power: float,
        power_analysis: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate human-readable interpretation of pretrends results."""
        
        interpretation = {
            "minimal_slope": f"Your pre-trends test would detect a linear trend with slope {minimal_slope:.4f} only {target_power*100}% of the time.",
            "power_assessment": "",
            "recommendation": ""
        }
        
        # Assess power level
        if target_power >= 0.8:
            if minimal_slope <= 0.01:
                interpretation["power_assessment"] = "Good power: Your test can detect very small trend violations."
                interpretation["recommendation"] = "Your pre-trends test appears to have adequate power for detecting meaningful violations."
            elif minimal_slope <= 0.05:
                interpretation["power_assessment"] = "Moderate power: Your test can detect modest trend violations."
                interpretation["recommendation"] = "Consider additional robustness checks or sensitivity analysis."
            else:
                interpretation["power_assessment"] = "Low power: Your test would only detect large trend violations."
                interpretation["recommendation"] = "Caution advised. Consider sensitivity analysis with HonestDiD to assess robustness to undetected violations."
        else:
            interpretation["power_assessment"] = f"Low target power ({target_power*100}%): Even detectable violations would be missed frequently."
            interpretation["recommendation"] = "Consider increasing sample size or using sensitivity analysis methods."
        
        # Add ex post analysis interpretation if available
        if power_analysis and "power_probability" in power_analysis:
            power_prob = power_analysis["power_probability"]
            if power_prob is not None:
                interpretation["ex_post"] = f"For the analyzed slope of {power_analysis['analyzed_slope']:.4f}, detection probability is {power_prob*100:.1f}%."
        
        return interpretation
    
    def dcdh_estimator(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        treatment_col: str,
        cohort_col: Optional[str] = None,
        mode: str = "dyn",
        effects: int = 5,
        placebo: int = 5,
        controls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        de Chaisemartin & D'Haultfoeuille estimator using DIDmultiplegt package.
        
        Flexible DID estimator that allows for treatment switching and 
        non-binary treatments, robust to heterogeneous treatment effects.
        
        Args:
            data: Panel data
            outcome_col: Outcome variable
            unit_col: Unit identifier
            time_col: Time variable
            treatment_col: Treatment variable (can be non-binary)
            cohort_col: Optional cohort/group variable
            mode: Estimation mode ("dyn", "stat", "had", "old")
            effects: Number of event-study effects to estimate
            placebo: Number of placebo effects to estimate
            controls: Optional list of control variables
            
        Returns:
            Dict with dCDH estimation results including event study
        """
        # Check for packages - prefer modern DIDmultiplegtDYN for dynamic mode
        if mode == "dyn" and 'DIDmultiplegtDYN' in self.r_packages:
            use_modern_dyn = True
            logger.info("Using modern DIDmultiplegtDYN package for dynamic effects")
        elif 'DIDmultiplegt' in self.r_packages:
            use_modern_dyn = False
            logger.info("Using legacy DIDmultiplegt package")
        else:
            return {
                "status": "error",
                "message": "DIDmultiplegt R package not available. Install with: install.packages('DIDmultiplegt')"
            }

        try:
            logger.info(f"Running de Chaisemartin & D'Haultfoeuille estimator in {mode} mode")

            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)

            # Build parameters based on mode
            if mode == "dyn" and use_modern_dyn:
                # Use modern did_multiplegt_dyn() from DIDmultiplegtDYN package
                logger.info(f"Running did_multiplegt_dyn with effects={effects}, placebo={placebo}")

                did_multiplegt_dyn = self.r_packages['DIDmultiplegtDYN'].did_multiplegt_dyn

                # Note: did_multiplegt_dyn expects numeric (not integer!) parameters
                # Pass Python int directly - rpy2 will convert to R numeric
                result = did_multiplegt_dyn(
                    r_data,
                    outcome=outcome_col,
                    group=unit_col,
                    time=time_col,
                    treatment=treatment_col,
                    effects=float(effects),  # Convert to float for R numeric
                    placebo=float(placebo),  # Convert to float for R numeric
                    graph_off=True
                )

            elif mode == "dyn":
                # Fallback to legacy method (will only return instantaneous effect)
                logger.warning("Using legacy did_multiplegt('dyn') - returns only instantaneous effect")
                did_multiplegt = self.r_packages['DIDmultiplegt'].did_multiplegt

                result = did_multiplegt(
                    "dyn",  # mode as first positional argument
                    r_data,  # data as second positional argument
                    outcome_col,  # Y
                    unit_col,    # G
                    time_col,    # T
                    treatment_col  # D
                )
            
            elif mode == "stat":
                # Static mode for instantaneous effects
                logger.info("Running static mode for instantaneous treatment effects")
                
                # Get did_multiplegt function if not already defined
                if 'did_multiplegt' not in locals():
                    did_multiplegt = self.r_packages['DIDmultiplegt'].did_multiplegt
                
                result = did_multiplegt(
                    "stat",  # mode as first positional argument
                    r_data,  # data as second positional argument
                    outcome_col,  # Y
                    unit_col,    # G
                    time_col,    # T
                    treatment_col,  # D
                    graph_off=True
                )
            
            else:
                # Other modes (had, old) for specialized cases
                logger.info(f"Running {mode} mode")
                
                # Get did_multiplegt function if not already defined
                if 'did_multiplegt' not in locals():
                    did_multiplegt = self.r_packages['DIDmultiplegt'].did_multiplegt
                
                result = did_multiplegt(
                    mode,  # mode as first positional argument
                    r_data,  # data as second positional argument
                    outcome_col,  # Y
                    unit_col,    # G
                    time_col,    # T
                    treatment_col,  # D
                    graph_off=True
                )
            
            logger.info("dCDH estimation completed, extracting results...")
            
            # Extract event study results
            event_study = {}
            overall_estimates = []
            overall_att_est = None
            att_se = None
            
            try:
                # Handle modern did_multiplegt_dyn() output (DIDmultiplegtDYN package)
                if mode == "dyn" and use_modern_dyn and hasattr(result, 'rx2'):
                    try:
                        # Extract effect table from did_multiplegt_dyn
                        # Structure: result$results$Effects
                        results_obj = result.rx2('results')
                        effect_obj = results_obj.rx2('Effects')

                        with localconverter(robjects.default_converter + pandas2ri.converter):
                            effect_array = robjects.conversion.rpy2py(effect_obj)

                        # effect_array is a numpy matrix with columns:
                        # Estimate, SE, LB CI, UB CI, N, Switchers
                        # Rows correspond to Effect_1, Effect_2, ..., Effect_N
                        logger.info(f"Extracted {len(effect_array)} dynamic effects from did_multiplegt_dyn")

                        # Build event study dictionary
                        # Period numbering: Effect_1 = period 0 (treatment period), Effect_2 = period 1, etc.
                        # Note: DIDmultiplegt_dyn defines Effect_1 as the FIRST period after treatment starts (t=0)
                        for idx, row in enumerate(effect_array):
                            period = idx  # Effect_1 → t=0, Effect_2 → t=1, ...
                            estimate = float(row[0])  # Estimate column
                            se = float(row[1])  # SE column

                            event_study[period] = {
                                "estimate": estimate,
                                "se": se,
                                "ci_lower": float(row[2]) if len(row) > 2 else estimate - 1.96 * se,
                                "ci_upper": float(row[3]) if len(row) > 3 else estimate + 1.96 * se,
                                "pvalue": 2 * (1 - robjects.r['pnorm'](float(abs(estimate / se)) if se > 0 else 0.0)[0]),
                                "n_obs": int(row[4]) if len(row) > 4 else None,
                                "n_switchers": int(row[5]) if len(row) > 5 else None
                            }

                            # Collect post-treatment estimates for overall ATT (periods >= 1)
                            overall_estimates.append(estimate)

                        # Extract Placebos if available
                        try:
                            placebo_obj = results_obj.rx2('Placebos')
                            with localconverter(robjects.default_converter + pandas2ri.converter):
                                placebo_array = robjects.conversion.rpy2py(placebo_obj)

                            logger.info(f"Extracted {len(placebo_array)} placebo tests")

                            # Placebos have negative periods
                            for idx, row in enumerate(placebo_array):
                                period = -(idx + 1)  # Placebo_1 = period -1, Placebo_2 = period -2
                                estimate = float(row[0])
                                se = float(row[1])

                                event_study[period] = {
                                    "estimate": estimate,
                                    "se": se,
                                    "ci_lower": float(row[2]) if len(row) > 2 else estimate - 1.96 * se,
                                    "ci_upper": float(row[3]) if len(row) > 3 else estimate + 1.96 * se,
                                    "pvalue": 2 * (1 - robjects.r['pnorm'](float(abs(estimate / se)) if se > 0 else 0.0)[0]),
                                    "n_obs": int(row[4]) if len(row) > 4 else None,
                                    "n_switchers": int(row[5]) if len(row) > 5 else None
                                }
                        except Exception as e:
                            logger.warning(f"Could not extract placebos: {e}")

                        # Extract official ATE for overall ATT (proper weighting)
                        try:
                            ate_obj = results_obj.rx2('ATE')
                            with localconverter(robjects.default_converter + pandas2ri.converter):
                                ate_array = robjects.conversion.rpy2py(ate_obj)

                            # ATE is a matrix (1 row, multiple columns): Estimate, SE, LB CI, UB CI, N, Switchers, ...
                            # Extract from first row
                            overall_att_est = float(ate_array[0, 0])  # Estimate column
                            att_se = float(ate_array[0, 1])  # SE column
                            logger.info(f"Extracted official ATE: {overall_att_est:.6f} (SE: {att_se:.6f})")
                        except Exception as e:
                            logger.warning(f"Could not extract ATE, using simple average: {e}")
                            # Fallback to simple average
                            if overall_estimates:
                                overall_att_est = np.mean(overall_estimates)
                                post_treatment_effects = {k: v for k, v in event_study.items() if k >= 1}
                                att_se = np.mean([v["se"] for v in post_treatment_effects.values()])
                                logger.info(f"Using simple average of {len(overall_estimates)} effects: {overall_att_est:.6f}")

                    except Exception as e:
                        logger.warning(f"Could not extract from did_multiplegt_dyn effect table: {e}")

                # Handle legacy did_multiplegt output (DIDmultiplegt package)
                elif mode == "dyn" and hasattr(result, 'rx2'):
                    # First, try to extract from coef (this contains the main estimates)
                    try:
                        coef_obj = result.rx2('coef')
                        if hasattr(coef_obj, 'rx2'):
                            # Extract estimate(s)
                            b_val = coef_obj.rx2('b')
                            vcov_val = coef_obj.rx2('vcov')
                            
                            # Convert to numpy for easier handling
                            with localconverter(robjects.default_converter + pandas2ri.converter):
                                b_array = robjects.conversion.rpy2py(b_val)
                                vcov_matrix = robjects.conversion.rpy2py(vcov_val)
                            
                            # Handle scalar or array
                            if np.isscalar(b_array) or (hasattr(b_array, 'shape') and b_array.shape == () or b_array.shape == (1,)):
                                # Single estimate
                                overall_att_est = float(b_array) if np.isscalar(b_array) else float(b_array.flat[0])
                                
                                # Extract SE from vcov (diagonal element or scalar)
                                if np.isscalar(vcov_matrix):
                                    att_se = float(np.sqrt(vcov_matrix))
                                elif hasattr(vcov_matrix, 'shape'):
                                    if vcov_matrix.shape == () or vcov_matrix.shape == (1,) or vcov_matrix.shape == (1, 1):
                                        att_se = float(np.sqrt(vcov_matrix.flat[0]))
                                    else:
                                        # Multiple periods - extract diagonal
                                        att_se = float(np.sqrt(np.diag(vcov_matrix)[0]))
                                
                                # Store as single period effect
                                event_study[0] = {
                                    "estimate": overall_att_est,
                                    "se": att_se,
                                    "ci_lower": overall_att_est - 1.96 * att_se,
                                    "ci_upper": overall_att_est + 1.96 * att_se,
                                    "pvalue": 2 * (1 - robjects.r['pnorm'](float(abs(overall_att_est / att_se)) if att_se > 0 else 0.0)[0])
                                }
                            else:
                                # Multiple estimates (event study)
                                for i in range(len(b_array)):
                                    estimate = float(b_array[i])
                                    se = float(np.sqrt(np.diag(vcov_matrix)[i])) if len(vcov_matrix.shape) > 1 else float(np.sqrt(vcov_matrix))
                                    
                                    event_study[i] = {
                                        "estimate": estimate,
                                        "se": se,
                                        "ci_lower": estimate - 1.96 * se,
                                        "ci_upper": estimate + 1.96 * se,
                                        "pvalue": 2 * (1 - robjects.r['pnorm'](float(abs(estimate / se)) if se > 0 else 0.0)[0])
                                    }
                                    if i >= 0:
                                        overall_estimates.append(estimate)
                                
                            logger.info(f"Extracted estimates from coef: ATT={overall_att_est}, SE={att_se}")
                    except Exception as e:
                        logger.warning(f"Could not extract from coef object: {e}")
                    
                    # Also check results component for additional info
                    results_obj = result.rx2('results')
                    if hasattr(results_obj, 'rx2'):
                        try:
                            # Effects matrix might have additional information
                            effects_data = results_obj.rx2('Effects')
                            # This is typically a matrix with columns: Estimate, SE, LB CI, UB CI, N, Switchers
                            # First row is the main effect
                            with localconverter(robjects.default_converter + pandas2ri.converter):
                                effects_array = robjects.conversion.rpy2py(effects_data)
                            
                            if hasattr(effects_array, 'shape') and effects_array.shape[1] >= 2:
                                # If we didn't get estimates from coef, use this
                                if overall_att_est is None:
                                    overall_att_est = float(effects_array[0, 0])  # Estimate
                                    att_se = float(effects_array[0, 1])  # SE
                                    
                                    event_study[0] = {
                                        "estimate": overall_att_est,
                                        "se": att_se,
                                        "ci_lower": float(effects_array[0, 2]) if effects_array.shape[1] > 2 else overall_att_est - 1.96 * att_se,
                                        "ci_upper": float(effects_array[0, 3]) if effects_array.shape[1] > 3 else overall_att_est + 1.96 * att_se,
                                        "pvalue": 2 * (1 - robjects.r['pnorm'](float(abs(overall_att_est / att_se)) if att_se > 0 else 0.0)[0])
                                    }
                        except Exception as e:
                            logger.warning(f"Could not extract Effects data: {e}")
                
                elif mode == "stat":
                    # Extract static ATT
                    if hasattr(result, "DID_l"):
                        estimate = float(result.rx2("DID_l")[0])
                        se = float(result.rx2("se_DID_l")[0]) if hasattr(result, "se_DID_l") else 0.0
                        
                        # Calculate p-value safely for static
                        if se > 0:
                            z_stat = float(abs(estimate / se))
                            pval = 2 * (1 - robjects.r['pnorm'](z_stat)[0])
                        else:
                            pval = 1.0
                            
                        event_study[0] = {
                            "estimate": estimate,
                            "se": se,
                            "ci_lower": estimate - 1.96 * se,
                            "ci_upper": estimate + 1.96 * se,
                            "pvalue": pval
                        }
                        overall_estimates.append(estimate)
                        
            except Exception as e:
                logger.warning(f"Could not extract some dCDH results: {e}")
            
            # Calculate overall ATT (if not already extracted)
            if overall_att_est is None:
                if overall_estimates:
                    overall_att_est = np.mean(overall_estimates)
                    # Use SE from first effect or average
                    att_se = event_study[0]["se"] if 0 in event_study else np.mean([v["se"] for v in event_study.values()]) if event_study else 0.0
                else:
                    overall_att_est = 0.0
                    att_se = 0.0
            
            # Ensure we have valid values
            if overall_att_est is None:
                overall_att_est = 0.0
            if att_se is None:
                att_se = 0.0
            
            # Calculate p-value safely for dcdh overall ATT
            if att_se > 0:
                z_stat = float(abs(overall_att_est / att_se))
                overall_pval = float(2 * (1 - robjects.r['pnorm'](z_stat)[0]))
            else:
                overall_pval = 1.0
                
            overall_att = {
                "estimate": float(overall_att_est),
                "se": float(att_se),
                "ci_lower": float(overall_att_est - 1.96 * att_se),
                "ci_upper": float(overall_att_est + 1.96 * att_se),
                "pvalue": overall_pval
            }
            
            # Extract number of observations
            try:
                n_obs = int(result.rx2("N")[0]) if hasattr(result, "N") else len(data)
            except:
                n_obs = len(data)
            
            return {
                "status": "success",
                "method": "de Chaisemartin & D'Haultfoeuille (2020, 2024)",
                "mode": mode,
                "overall_att": overall_att,
                "event_study": event_study,
                "n_periods": len(event_study),
                "n_obs": n_obs,
                "allows_treatment_switching": True,
                "allows_continuous_treatment": True
            }
            
        except Exception as e:
            logger.error(f"Error in dCDH estimation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def efficient_estimator(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        cohort_col: str,
        estimand: str = "simple",
        event_time: Optional[List[int]] = None,
        use_cs: bool = False,
        use_sa: bool = False,
        beta: Optional[float] = 1.0
    ) -> Dict[str, Any]:
        """
        Roth & Sant'Anna (2023) efficient estimator using staggered package.

        Computes the efficient estimator for randomized staggered rollout designs,
        potentially offering substantial gains over parallel-trends-only methods.

        ⚠️  IMPORTANT - Small Sample Warning:
        =====================================
        The plug-in efficient estimator (beta=None) is ASYMPTOTICALLY unbiased but may
        exhibit SUBSTANTIAL FINITE SAMPLE BIAS, especially with n < 500.

        Empirical testing shows:
        - Small samples (n < 500): Plug-in beta can produce incorrect estimates (even wrong sign!)
        - Large samples (n > 1000): Plug-in beta offers superior efficiency

        RECOMMENDATION:
        - For small/moderate samples: Use beta=1.0 (Callaway & Sant'Anna, DEFAULT)
        - For large samples (n > 1000): Consider beta=None for efficiency gains
        - Always validate results by comparing with estimate_callaway_santanna()

        SUPPORTED ESTIMANDS:
        ===================
        - estimand="simple": Overall ATT with group-size weights (RECOMMENDED)
        - estimand="cohort": Cohort-specific treatment effects
        - estimand="calendar": Calendar-time-specific treatment effects

        NOT SUPPORTED:
        - estimand="eventstudy": Use estimate_callaway_santanna(), estimate_sun_abraham(),
          or estimate_bjs_imputation() for event study analysis

        Args:
            data: Panel data
            outcome_col: Outcome variable
            unit_col: Unit identifier
            time_col: Time variable
            cohort_col: Treatment timing group variable (first treatment time)
            estimand: Aggregation type ("simple", "cohort", or "calendar")
            event_time: DEPRECATED - Not used (eventstudy not supported)
            use_cs: Use Callaway & Sant'Anna weighting (sets beta=1, use_last_treated_only=False)
            use_sa: Use Sun & Abraham weighting (sets beta=1, use_last_treated_only=True)
            beta: Weight parameter for efficiency (ignored if use_cs or use_sa is True)
                  - 1.0 (DEFAULT): Callaway & Sant'Anna weighting - ROBUST in finite samples
                  - None: Plug-in efficient beta - USE ONLY for large samples (n > 1000)
                  - 0: Simple difference-in-means

        Returns:
            Dict containing:
            - status: "success" or "error"
            - method: Method description
            - estimand: Requested estimand type
            - overall_att: Dict with estimate, se, ci_lower, ci_upper, pvalue, source
              * source indicates which estimand was used ("simple_estimand" or "eventstudy_via_simple")
            - event_study: Dict of {event_time: {estimate, se, ci_lower, ci_upper, pvalue}}
            - n_periods: Number of event time periods
            - beta: Efficiency parameter used
            - neyman_se: Neyman standard error (if available)
            - notes: Additional notes about the estimation

        Examples:
            >>> # Get overall ATT using simple estimand (RECOMMENDED)
            >>> result = estimator.efficient_estimator(
            ...     data, "outcome", "unit", "time", "cohort", estimand="simple"
            ... )
            >>> print(result['overall_att'])  # Single aggregate estimate

            >>> # Get both overall ATT and event study (AUTO-DUAL-ESTIMATION)
            >>> result = estimator.efficient_estimator(
            ...     data, "outcome", "unit", "time", "cohort",
            ...     estimand="eventstudy", event_time=[-3, -2, -1, 0, 1, 2, 3]
            ... )
            >>> print(result['overall_att'])  # From "simple" estimand
            >>> print(result['event_study'])  # From "eventstudy" estimand
            >>> print(result['notes'])        # Explains dual estimation
        """
        if not R_AVAILABLE or 'staggered' not in self.r_packages:
            return {
                "status": "error",
                "message": "staggered R package not available. Install with: install.packages('staggered')"
            }
        
        try:
            logger.info(f"Running Roth & Sant'Anna efficient estimator with estimand={estimand}")
            
            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)
            
            # ALWAYS use main staggered() function (not wrapper functions)
            # Wrapper functions (staggered_cs, staggered_sa) don't accept beta or use_last_treated_only
            staggered_func = self.r_packages['staggered'].staggered

            # Set parameters based on use_cs/use_sa flags
            if use_cs:
                # Callaway & Sant'Anna: beta=1, use_last_treated_only=False
                beta = 1.0
                use_last_treated_only = False
                logger.info("Using Callaway & Sant'Anna weighting (beta=1, use_last_treated_only=False)")
            elif use_sa:
                # Sun & Abraham: beta=1, use_last_treated_only=True
                beta = 1.0
                use_last_treated_only = True
                logger.info("Using Sun & Abraham weighting (beta=1, use_last_treated_only=True)")
            else:
                # Default: use specified beta (or None for optimal), use_last_treated_only=False
                use_last_treated_only = False
                if beta is None:
                    logger.info("Using optimal plug-in efficient beta (beta=NULL, use_last_treated_only=False)")
                else:
                    logger.info(f"Using beta={beta}, use_last_treated_only=False")

            # Convert Python None to R NULL for beta parameter
            r_beta = robjects.NULL if beta is None else beta

            # Validate estimand parameter
            if estimand == "eventstudy":
                return {
                    "status": "error",
                    "message": (
                        "estimand='eventstudy' is not supported for the efficient estimator. "
                        "The R staggered package's eventstudy implementation has technical limitations "
                        "(requires eventTime parameter which causes errors on short panels). "
                        "\n\nFor event study analysis, please use other methods:\n"
                        "- estimate_callaway_santanna() - Robust event study with clean 2x2 comparisons\n"
                        "- estimate_sun_abraham() - Fast interaction-weighted estimator\n"
                        "- estimate_bjs_imputation() - Imputation-based event study\n"
                        "\nFor overall ATT with efficient estimator, use estimand='simple' (recommended)."
                    )
                }

            event_study = {}
            overall_att_est = 0.0
            att_se = 0.0
            estimation_notes = []

            # Single estimation for supported aggregations (simple, cohort, calendar)
            logger.info(f"Running '{estimand}' estimand")
            result = staggered_func(
                df=r_data,
                i=unit_col,
                t=time_col,
                g=cohort_col,
                y=outcome_col,
                estimand=estimand,
                beta=r_beta,
                use_last_treated_only=use_last_treated_only
            )

            estimation_notes.append(f"Single estimation using '{estimand}' estimand")

            logger.info("Efficient estimation completed, extracting results...")

            # Convert result to dataframe for extraction
            with localconverter(robjects.default_converter + pandas2ri.converter):
                result_df = robjects.conversion.rpy2py(result)

            # Process results (only simple/cohort/calendar supported)
            estimate = float(result_df.iloc[0]['estimate'])
            se = float(result_df.iloc[0]['se'])

            overall_att_est = estimate
            att_se = se

            # Store as single period for consistency
            if se > 0:
                z_stat = float(abs(estimate / se))
                pval = 2 * (1 - robjects.r['pnorm'](z_stat)[0])
            else:
                pval = 1.0

            event_study[0] = {
                "estimate": estimate,
                "se": se,
                "ci_lower": estimate - 1.96 * se,
                "ci_upper": estimate + 1.96 * se,
                "pvalue": pval
            }
            
            # Calculate overall p-value safely for efficient estimator
            if att_se > 0:
                z_stat = float(abs(overall_att_est / att_se))
                overall_pval = float(2 * (1 - robjects.r['pnorm'](z_stat)[0]))
            else:
                overall_pval = 1.0
                
            # Determine source for overall_att
            att_source = f"{estimand}_estimand"

            overall_att = {
                "estimate": float(overall_att_est),
                "se": float(att_se),
                "ci_lower": float(overall_att_est - 1.96 * att_se),
                "ci_upper": float(overall_att_est + 1.96 * att_se),
                "pvalue": overall_pval,
                "source": att_source
            }

            # Get Neyman SE if available
            neyman_se = None
            if 'se_neyman' in result_df.columns:
                neyman_se = float(result_df.iloc[0]['se_neyman'])

            return {
                "status": "success",
                "method": f"Roth & Sant'Anna (2023) Efficient Estimator",
                "estimand": estimand,
                "beta": beta,
                "overall_att": overall_att,
                "event_study": event_study,
                "n_periods": len(event_study),
                "neyman_se": neyman_se,
                "efficiency_gain": "Potentially substantial over parallel-trends-only methods",
                "notes": " | ".join(estimation_notes)
            }
            
        except Exception as e:
            logger.error(f"Error in efficient estimation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def gsynth_estimator(
        self,
        data: pd.DataFrame,
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
        nboots: int = 200,
        min_T0: Optional[int] = None,
        parallel: bool = True,
        cores: int = 4
    ) -> Dict[str, Any]:
        """
        Generalized Synthetic Control Method (Xu 2017).

        Uses interactive fixed effects model to estimate treatment effects
        when parallel trends may be violated due to unobserved time-varying
        confounders.

        Reference:
            Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference
            with Interactive Fixed Effects Models." Political Analysis, 25(1), 57-76.

        Args:
            data: Panel data
            outcome_col: Outcome variable name
            unit_col: Unit identifier name
            time_col: Time variable name
            treatment_col: Binary treatment indicator (0/1)
            covariates: List of time-varying covariate names
            force: Fixed effects - "none", "unit", "time", or "two-way"
            CV: Use cross-validation to select number of factors
            r_range: Tuple (min, max) for number of factors to consider
            se: Compute standard errors
            inference: "parametric" or "nonparametric" (bootstrap)
            nboots: Number of bootstrap replications if inference="nonparametric"
            min_T0: Minimum number of pre-treatment periods required
            parallel: Use parallel processing
            cores: Number of cores for parallel processing

        Returns:
            Dict with:
            - status: "success" or "error"
            - method: "Generalized Synthetic Control (Xu 2017)"
            - overall_att: Average treatment effect on treated
            - att_by_period: Treatment effects by post-treatment period
            - n_factors: Number of factors selected
            - pre_treatment_fit: Pre-treatment MSPE
            - diagnostics: Model fit information

        Notes:
            - Handles staggered adoption automatically
            - Suitable for small N, large T panels
            - Requires at least min_T0 pre-treatment periods (default: 5)
            - Cross-validation recommended for factor selection
        """
        if not R_AVAILABLE or 'gsynth' not in self.r_packages:
            return {
                "status": "error",
                "message": "gsynth R package not available. Install with: install.packages('gsynth')"
            }

        try:
            logger.info(f"Running Generalized Synthetic Control (gsynth)")

            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)

            # Build formula
            if covariates:
                covariate_str = " + " + " + ".join(covariates)
            else:
                covariate_str = ""
            formula_str = f"{outcome_col} ~ {treatment_col}{covariate_str}"
            logger.info(f"Formula: {formula_str}")

            # Get gsynth function
            gsynth_fn = self.r_packages['gsynth'].gsynth

            # Build arguments
            gsynth_args = {
                "formula": robjects.Formula(formula_str),
                "data": r_data,
                "index": robjects.StrVector([unit_col, time_col]),
                "force": force,
                "CV": CV,
                "r": robjects.IntVector(list(range(r_range[0], r_range[1] + 1))),
                "se": se,
                "inference": inference,
                "nboots": nboots,
                "parallel": parallel,
                "cores": cores
            }

            if min_T0 is not None:
                gsynth_args["min_T0"] = min_T0

            # Run gsynth
            logger.info(f"Running gsynth with CV={CV}, force={force}")
            gsynth_result = gsynth_fn(**gsynth_args)

            # Extract results
            result = {
                "status": "success",
                "method": "Generalized Synthetic Control (Xu 2017)",
                "formula": formula_str,
                "force": force,
                "inference": inference
            }

            # Extract ATT from est.avg matrix
            # est.avg is a 1x5 matrix: [Estimate, S.E., CI.lower, CI.upper, p.value]
            est_avg_matrix = gsynth_result.rx2('est.avg')

            # Extract estimate (first column)
            att_estimate = float(est_avg_matrix[0])

            # Extract standard error and confidence intervals
            att_se = None
            ci_lower = None
            ci_upper = None
            p_value = None

            if se:
                try:
                    # est.avg is a matrix, extract columns by index
                    # Column 0: Estimate, Column 1: S.E., Column 2: CI.lower, Column 3: CI.upper, Column 4: p.value
                    att_se = float(est_avg_matrix[1]) if len(est_avg_matrix) > 1 else None
                    ci_lower = float(est_avg_matrix[2]) if len(est_avg_matrix) > 2 else None
                    ci_upper = float(est_avg_matrix[3]) if len(est_avg_matrix) > 3 else None
                    p_value = float(est_avg_matrix[4]) if len(est_avg_matrix) > 4 else None
                except Exception as e:
                    logger.warning(f"Could not extract inference statistics from est.avg: {e}")
                    # Fallback to manual calculation if extraction fails
                    if att_se is None:
                        att_se = None
                        ci_lower = None
                        ci_upper = None

            result["overall_att"] = {
                "estimate": att_estimate,
                "se": att_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
            }

            # Extract period-by-period ATT
            att_by_period = {}
            att_est = gsynth_result.rx2('att')
            if att_est is not None:
                periods = list(att_est.names)
                for i, period in enumerate(periods):
                    att_by_period[period] = {
                        "estimate": float(list(att_est)[i])
                    }
            result["att_by_period"] = att_by_period

            # Extract model information with safe access
            try:
                if CV:
                    r_cv = gsynth_result.rx2('r.cv')
                    result["n_factors"] = int(r_cv[0]) if r_cv is not robjects.NULL else r_range[1]
                else:
                    result["n_factors"] = r_range[1]
            except:
                result["n_factors"] = r_range[1]

            try:
                n_t = gsynth_result.rx2('N.t')
                result["n_treated"] = int(n_t[0]) if n_t is not robjects.NULL else None
            except:
                result["n_treated"] = None

            try:
                n_co = gsynth_result.rx2('N.co')
                result["n_control"] = int(n_co[0]) if n_co is not robjects.NULL else None
            except:
                result["n_control"] = None

            try:
                t_val = gsynth_result.rx2('T')
                result["n_periods"] = int(t_val[0]) if t_val is not robjects.NULL else None
            except:
                result["n_periods"] = None

            # Pre-treatment fit
            try:
                mspe = gsynth_result.rx2('pre.sd')
                if mspe is not None and mspe is not robjects.NULL:
                    result["pre_treatment_mspe"] = float(mspe[0])
            except:
                pass

            logger.info(f"gsynth complete: ATT = {result['overall_att']['estimate']:.4f}, factors = {result['n_factors']}")
            return result

        except Exception as e:
            logger.error(f"Error in gsynth estimator: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"gsynth estimation failed: {str(e)}"
            }

    def synthdid_estimator(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        treatment_col: str,
        cohort_col: Optional[str] = None,
        vcov_method: str = "placebo",
        weights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthetic Difference-in-Differences estimator (Arkhangelsky et al. 2019).

        Combines synthetic control method with difference-in-differences by
        estimating both unit and time weights to construct a synthetic control.

        Reference:
            Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
            (2019). "Synthetic Difference in Differences." NBER Working Paper 25532.

        Args:
            data: Panel data
            outcome_col: Outcome variable name
            unit_col: Unit identifier name
            time_col: Time variable name
            treatment_col: Binary treatment indicator (0/1)
            cohort_col: Optional cohort variable (for staggered adoption)
            vcov_method: Variance estimation method - "placebo", "bootstrap", or "jackknife"
            weights: Optional dictionary with "lambda" and "omega" for custom weights

        Returns:
            Dict with:
            - status: "success" or "error"
            - method: "Synthetic Difference-in-Differences (Arkhangelsky et al. 2019)"
            - overall_att: Treatment effect estimate
            - unit_weights: Weights assigned to control units
            - time_weights: Weights assigned to pre-treatment periods
            - comparison_methods: Results from traditional DID and SC for comparison

        Important Notes:
            - **Requires all treated units to begin treatment simultaneously**
            - If staggered adoption, must convert to cohort-specific analysis
            - Package currently in beta, interface may change
        """
        if not R_AVAILABLE or 'synthdid' not in self.r_packages:
            return {
                "status": "error",
                "message": "synthdid R package not available. Install with: devtools::install_github('synth-inference/synthdid')"
            }

        try:
            logger.info(f"Running Synthetic Difference-in-Differences (synthdid)")

            # Check for simultaneous treatment timing
            if cohort_col:
                cohorts = data[cohort_col].unique()
                if len(cohorts) > 2:  # More than never-treated (0) and one treatment cohort
                    logger.warning(f"synthdid works best with simultaneous treatment. Found {len(cohorts)-1} treatment cohorts.")

            # Convert to R dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_data = robjects.conversion.py2rpy(data)

            # Get synthdid functions
            panel_matrices = self.r_packages['synthdid'].panel_matrices
            synthdid_estimate = self.r_packages['synthdid'].synthdid_estimate

            # Prepare panel matrices
            # synthdid's panel_matrices expects: panel, unit, time, outcome, treatment
            # where unit/time/outcome/treatment are column names or indices
            setup = panel_matrices(
                panel=r_data,
                unit=unit_col,
                time=time_col,
                outcome=outcome_col,
                treatment=treatment_col
            )

            # Extract Y, N0, T0 from setup
            Y = setup.rx2('Y')
            N0 = setup.rx2('N0')
            T0 = setup.rx2('T0')

            # Estimate SDID
            logger.info("Computing Synthetic DiD estimate")
            tau_sdid = synthdid_estimate(Y, N0, T0)

            # Compute standard error using vcov() (official API)
            # Note: bootstrap and jackknife methods return NA when N_treated = 1
            logger.info(f"Computing standard error with method: {vcov_method}")
            vcov_result = robjects.r['vcov'](tau_sdid, method=vcov_method)
            se_sdid = float(robjects.r['sqrt'](vcov_result)[0])

            # Check if SE is NA (happens when N_treated = 1 for bootstrap/jackknife)
            if np.isnan(se_sdid):
                # Count treated units from R objects
                Y_matrix = setup.rx2('Y')
                n_control = int(setup.rx2('N0')[0])
                n_total = int(robjects.r['nrow'](Y_matrix)[0])
                n_treated = n_total - n_control

                if n_treated == 1 and vcov_method in ['bootstrap', 'jackknife']:
                    error_msg = (
                        f"Variance method '{vcov_method}' is mathematically undefined when there is only 1 treated unit.\n\n"
                        f"Why this happens:\n"
                        f"  - Bootstrap requires resampling multiple treated units (N_treated ≥ 2)\n"
                        f"  - Jackknife requires leaving out units, but becomes undefined when removing the only treated unit\n\n"
                        f"Solution:\n"
                        f"  Use vcov_method='placebo' instead, which works correctly for single treated units.\n\n"
                        f"Your data:\n"
                        f"  - Treated units: {n_treated}\n"
                        f"  - Control units: {n_control}\n"
                        f"  - Total units: {n_total}\n\n"
                        f"Reference: Arkhangelsky et al. (2021) 'Synthetic Difference in Differences', Section 5"
                    )
                    logger.error(error_msg)
                    return {
                        "status": "error",
                        "message": error_msg
                    }
                else:
                    # Other reason for NA
                    logger.warning(f"Standard error is NA for method '{vcov_method}', using None")

            # Also compute traditional DiD and SC for comparison
            logger.info("Computing traditional DiD and Synthetic Control for comparison")
            did_estimate = self.r_packages['synthdid'].did_estimate
            sc_estimate = self.r_packages['synthdid'].sc_estimate

            tau_did = did_estimate(Y, N0, T0)
            tau_sc = sc_estimate(Y, N0, T0)

            # Extract results
            result = {
                "status": "success",
                "method": "Synthetic Difference-in-Differences (Arkhangelsky et al. 2019)",
                "vcov_method": vcov_method
            }

            # Main estimate
            result["overall_att"] = {
                "estimate": float(tau_sdid[0]),
                "se": float(se_sdid),
                "ci_lower": float(tau_sdid[0] - 1.96 * se_sdid),
                "ci_upper": float(tau_sdid[0] + 1.96 * se_sdid),
            }

            # Comparison with traditional methods
            result["comparison_methods"] = {
                "traditional_did": {
                    "estimate": float(tau_did[0]),
                    "note": "Traditional difference-in-differences (equal weights)"
                },
                "synthetic_control": {
                    "estimate": float(tau_sc[0]),
                    "note": "Traditional synthetic control (no time weights)"
                },
                "synthdid": {
                    "estimate": float(tau_sdid[0]),
                    "note": "Synthetic DiD (unit + time weights)"
                }
            }

            # Extract weights
            try:
                weights_obj = robjects.r['attr'](tau_sdid, 'weights')
                if weights_obj is not None:
                    omega = weights_obj.rx2('omega')  # Unit weights
                    lambda_w = weights_obj.rx2('lambda')  # Time weights

                    result["unit_weights"] = {
                        "n_nonzero": int(robjects.r['sum'](omega > 0)[0]),
                        "max_weight": float(robjects.r['max'](omega)[0])
                    }
                    result["time_weights"] = {
                        "n_nonzero": int(robjects.r['sum'](lambda_w > 0)[0]),
                        "max_weight": float(robjects.r['max'](lambda_w)[0])
                    }
            except Exception as e:
                logger.warning(f"Could not extract weights: {e}")

            # Sample size information
            result["n_treated_units"] = int(Y.nrow) - int(N0[0])
            result["n_control_units"] = int(N0[0])
            result["n_pretreatment_periods"] = int(T0[0])
            result["n_posttreatment_periods"] = int(Y.ncol) - int(T0[0])

            logger.info(f"synthdid complete: ATT = {result['overall_att']['estimate']:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error in synthdid estimator: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"synthdid estimation failed: {str(e)}"
            }