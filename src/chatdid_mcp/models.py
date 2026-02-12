"""
Pydantic models for parameter validation where FastMCP's built-in
type-annotation validation is insufficient (e.g., complex parsing logic).

Most MCP tool parameters are validated directly via function signatures
and FastMCP's type coercion. Pydantic models are only used when a tool
needs non-trivial validation beyond what type hints provide.
"""

from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator
import json


class SensitivityAnalysisParams(BaseModel):
    """Parameters for HonestDiD sensitivity analysis.

    This model exists because m_values requires non-trivial parsing:
    the MCP client may send it as a JSON array, a string representation
    of an array, or omit it entirely. A plain type hint cannot express this.

    Handles flexible input formats for m_values parameter:
    - JSON array: [0.5, 1.0, 1.5, 2.0]
    - String representation: "[0.5, 1.0, 1.5, 2.0]"
    - None (uses default)
    """
    data_id: str = Field("current", description="Dataset identifier")
    method: Literal["relative_magnitude", "smoothness"] = Field(
        "relative_magnitude",
        description="Sensitivity analysis method"
    )
    m_values: Optional[Union[List[float], str]] = Field(
        None,
        description="M values for sensitivity (list of floats or JSON string)"
    )
    event_time: int = Field(0, description="Target event time for analysis")
    confidence_level: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for intervals"
    )
    estimator_method: Literal["callaway_santanna", "sun_abraham"] = Field(
        "callaway_santanna",
        description="Base DID estimator"
    )

    @field_validator('m_values')
    @classmethod
    def parse_m_values(cls, v):
        """Parse m_values from string if needed."""
        if v is None:
            return None

        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    if all(isinstance(x, (int, float)) for x in parsed):
                        return parsed
                    else:
                        raise ValueError("All m_values must be numeric")
                else:
                    raise ValueError("m_values string must represent a JSON array")
            except json.JSONDecodeError as e:
                raise ValueError(f"Cannot parse m_values as JSON: {e}") from e

        if isinstance(v, list):
            if all(isinstance(x, (int, float)) for x in v):
                return v
            else:
                raise ValueError("All m_values must be numeric")

        raise ValueError(f"m_values must be a list or JSON string, got {type(v)}")

    @field_validator('confidence_level')
    @classmethod
    def validate_confidence_level(cls, v):
        """Validate confidence level is between 0 and 1."""
        if not (0 < v < 1):
            raise ValueError("Confidence level must be between 0 and 1")
        return v
