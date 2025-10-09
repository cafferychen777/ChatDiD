"""
Pydantic models for parameter validation
Ensures robust input validation following MCP best practices
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class DiDAnalysisConfig(BaseModel):
    """Configuration for DID analysis"""
    unit_col: str = Field(..., min_length=1, description="Unit identifier column")
    time_col: str = Field(..., min_length=1, description="Time variable column")
    outcome_col: str = Field(..., min_length=1, description="Outcome variable column")
    treatment_col: str = Field(..., min_length=1, description="Treatment indicator column")
    cohort_col: Optional[str] = Field(None, description="Treatment cohort column")


class DiDWorkflowParams(BaseModel):
    """Parameters for complete DID workflow"""
    unit_col: str = Field(..., min_length=1)
    time_col: str = Field(..., min_length=1)
    outcome_col: str = Field(..., min_length=1)
    treatment_col: str = Field(..., min_length=1)
    cohort_col: Optional[str] = None
    method: Literal[
        "auto",
        "callaway_santanna",
        "sun_abraham",
        "imputation_bjs",
        "gardner",
        "dcdh",
        "efficient",
        "gsynth",
        "synthdid"
    ] = Field("auto", description="Estimation method")
    cluster_level: Optional[str] = Field(None, description="Clustering variable")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        """Validate estimation method"""
        valid_methods = [
            "auto", "callaway_santanna", "sun_abraham",
            "imputation_bjs", "gardner", "dcdh", "efficient",
            "gsynth", "synthdid"
        ]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")
        return v


class GoodmanBaconParams(BaseModel):
    """Parameters for Goodman-Bacon decomposition"""
    formula: str = Field(..., min_length=3, description="R formula (e.g., 'y ~ d')")
    id_var: str = Field(..., min_length=1, description="Unit identifier")
    time_var: str = Field(..., min_length=1, description="Time variable")
    
    @field_validator('formula')
    @classmethod
    def validate_formula(cls, v):
        """Validate R formula format"""
        if '~' not in v:
            raise ValueError("Formula must contain '~' (e.g., 'outcome ~ treatment')")
        return v


class CallawySantAnnaParams(BaseModel):
    """Enhanced parameters for Callaway & Sant'Anna estimator"""
    # Required parameters
    yname: str = Field(..., min_length=1, description="Outcome variable")
    tname: str = Field(..., min_length=1, description="Time variable")
    idname: str = Field(..., min_length=1, description="Unit ID")
    gname: str = Field(..., min_length=1, description="Group/cohort variable")
    
    # Control group specification
    control_group: Literal["notyettreated", "nevertreated"] = Field(
        "notyettreated",
        description="Control group type"
    )
    
    # Covariates and anticipation
    xformla: Optional[str] = Field(None, description="Covariate formula")
    anticipation: int = Field(0, ge=0, description="Anticipation periods")
    
    # NEW: Estimation method parameters
    est_method: Literal["dr", "ipw", "reg"] = Field(
        "dr", 
        description="Estimation method: dr (doubly robust), ipw (inverse probability weighting), reg (regression)"
    )
    
    # NEW: Clustering and inference
    clustervars: Optional[str] = Field(None, description="Variables for clustering standard errors")
    bstrap: bool = Field(True, description="Use bootstrap for inference")
    biters: int = Field(1000, ge=100, le=10000, description="Bootstrap iterations")
    alp: float = Field(0.05, gt=0, lt=1, description="Significance level for confidence intervals")
    cband: bool = Field(True, description="Compute uniform confidence bands")
    
    # NEW: Sample weights and panel structure
    weightsname: Optional[str] = Field(None, description="Variable name for sample weights")
    panel: bool = Field(True, description="Is data panel (vs repeated cross-sections)")
    allow_unbalanced_panel: bool = Field(False, description="Allow unbalanced panel data")
    
    # NEW: Base period for comparisons
    base_period: Literal["varying", "universal"] = Field(
        "varying", 
        description="Base period for comparisons"
    )
    
    # NEW: Aggregation type for results
    aggregation_type: Literal["dynamic", "group", "calendar", "simple"] = Field(
        "dynamic",
        description="Type of aggregation: dynamic (event study), group (by cohort), calendar (by time), simple (overall ATT)"
    )
    
    @field_validator('xformla')
    @classmethod
    def validate_xformla(cls, v):
        """Validate covariate formula if provided"""
        if v and '~' not in v:
            raise ValueError("Covariate formula must start with '~' (e.g., '~ X1 + X2')")
        return v
    
    @field_validator('biters')
    @classmethod
    def validate_biters(cls, v):
        """Validate bootstrap iterations"""
        if v < 100:
            raise ValueError("Bootstrap iterations should be at least 100 for reliable inference")
        return v


class DataLoadParams(BaseModel):
    """Parameters for data loading"""
    file_path: str = Field(..., min_length=1, description="Path to data file")
    file_type: Literal["auto", "csv", "xlsx", "xls", "dta", "parquet"] = Field(
        "auto",
        description="File type"
    )
    sheet_name: Optional[str] = Field(None, description="Excel sheet name")
    encoding: str = Field("utf-8", description="File encoding")


class DiagnosticParams(BaseModel):
    """Parameters for TWFE diagnostics"""
    run_bacon_decomp: bool = Field(True, description="Run Goodman-Bacon decomposition")
    run_twfe_weights: bool = Field(True, description="Analyze TWFE weights")
    check_negative_weights: bool = Field(True, description="Check for negative weights")
    
    
class EstimationParams(BaseModel):
    """Parameters for DID estimation"""
    method: str = Field(..., description="Estimation method")
    control_group: Literal["notyettreated", "nevertreated"] = Field(
        "notyettreated",
        description="Control group for CS estimator"
    )
    covariates: List[str] = Field(default_factory=list, description="Covariate list")
    cluster_var: Optional[str] = Field(None, description="Clustering variable")
    event_study: bool = Field(True, description="Generate event study")
    min_e: int = Field(-10, description="Minimum event time")
    max_e: int = Field(10, description="Maximum event time")