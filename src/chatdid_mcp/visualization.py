"""
Visualization tools for ChatDiD MCP server.
Implements event study plots and diagnostic plots for DID analysis.
"""

from __future__ import annotations  # For forward type references

import logging
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
import numpy as np
from datetime import datetime
import base64
import io
from pathlib import Path

if TYPE_CHECKING:
    from mcp.types import ImageContent

logger = logging.getLogger(__name__)

# Import FastMCP image types for Claude Desktop display
try:
    from fastmcp.utilities.types import Image
    from mcp.types import ImageContent
    FASTMCP_IMAGE_AVAILABLE = True
except ImportError:
    FASTMCP_IMAGE_AVAILABLE = False
    logger.warning("FastMCP image types not available - inline display will be disabled")

# Import the storage manager
try:
    from .storage_manager import StorageManager
except ImportError:
    # If running as standalone script
    from storage_manager import StorageManager

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib backend loaded successfully")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(f"Matplotlib not available: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    logger.info("Plotly backend loaded successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    logger.warning(f"Plotly not available: {e}")


class DiDVisualizer:
    """
    Main visualization class for DID analysis.
    Supports both matplotlib and plotly backends.
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize visualizer with specified backend.
        
        Args:
            backend: "matplotlib" or "plotly"
        """
        self.backend = backend
        
        # Initialize storage manager for file handling
        self.storage_manager = StorageManager()
        
        # Keep figures_dir for backward compatibility
        self.figures_dir = str(self.storage_manager.base_dir)
        
        # Set style preferences
        if backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif backend == "plotly" and PLOTLY_AVAILABLE:
            pio.templates.default = "plotly_white"
        
        logger.info(f"DiDVisualizer initialized with {backend} backend")
    
    def create_event_study_plot(
        self,
        results: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_pretrends: bool = True,
        confidence_level: float = 0.95,
        display_mode: str = "both",
        max_inline_size: int = 1_000_000,
        auto_optimize: bool = True
    ) -> Union[Dict[str, Any], ImageContent]:
        """
        Create event study plot from DID estimation results.

        Args:
            results: DID estimation results with event_study data
            title: Plot title
            save_path: Path to save the plot
            show_pretrends: Whether to highlight pre-treatment periods
            confidence_level: Confidence level for intervals
            display_mode: How to return the plot
                - "display": Return ImageContent only (for inline display)
                - "save": Save to file and return metadata only
                - "both": Save file + return ImageContent if size permits (default)
            max_inline_size: Maximum size for inline display (default: 1MB)
            auto_optimize: Automatically compress if needed (default: True)

        Returns:
            Dict with plot info, or ImageContent for display, depending on display_mode
        """
        if "event_study" not in results:
            return {
                "status": "error",
                "message": "No event study data found in results"
            }
        
        event_study = results["event_study"]
        method = results.get("method", "DID Estimator")
        
        # Extract data
        event_times = sorted(event_study.keys())
        estimates = [event_study[t]["estimate"] for t in event_times]
        ses = [event_study[t]["se"] for t in event_times]
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        ci_lower = [est - z_score * se for est, se in zip(estimates, ses, strict=False)]
        ci_upper = [est + z_score * se for est, se in zip(estimates, ses, strict=False)]
        
        if self.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_event_study(
                event_times, estimates, ci_lower, ci_upper,
                method, title, save_path, show_pretrends, confidence_level,
                display_mode, max_inline_size, auto_optimize
            )
        elif self.backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_event_study(
                event_times, estimates, ci_lower, ci_upper,
                method, title, save_path, show_pretrends, confidence_level,
                display_mode, max_inline_size, auto_optimize
            )
        else:
            return {
                "status": "error",
                "message": f"Backend {self.backend} not available"
            }
    
    def _create_matplotlib_event_study(
        self,
        event_times: List[int],
        estimates: List[float],
        ci_lower: List[float],
        ci_upper: List[float],
        method: str,
        title: Optional[str],
        save_path: Optional[str],
        show_pretrends: bool,
        confidence_level: float,
        display_mode: str = "both",
        max_inline_size: int = 1_000_000,
        auto_optimize: bool = True
    ) -> Union[Dict[str, Any], ImageContent]:
        """Create event study plot using matplotlib."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot confidence intervals
        ax.fill_between(event_times, ci_lower, ci_upper, 
                       alpha=0.3, color='lightblue', 
                       label=f'{int(confidence_level*100)}% Confidence Interval')
        
        # Plot point estimates
        ax.plot(event_times, estimates, 'o-', color='darkblue', 
               linewidth=2, markersize=6, label='Point Estimates')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
        
        # Add vertical line at treatment time (event time 0)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, label='Treatment Start')
        
        # Highlight pre-treatment period if requested
        if show_pretrends:
            pre_treatment_times = [t for t in event_times if t < 0]
            if pre_treatment_times:
                ax.axvspan(min(pre_treatment_times), 0, alpha=0.1, color='gray', 
                          label='Pre-treatment Period')
        
        # Formatting
        ax.set_xlabel('Event Time (Periods Relative to Treatment)', fontsize=12)
        ax.set_ylabel('Treatment Effect', fontsize=12)
        ax.set_title(title or f'Event Study Plot - {method}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add method info
        ax.text(0.02, 0.98, f'Method: {method}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()

        # Convert to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        plt.close()

        # Optimize size if needed
        original_size = len(image_bytes)
        if auto_optimize and original_size > max_inline_size:
            image_bytes = self._optimize_image_size(image_bytes, max_inline_size)
            logger.info(f"Image optimized: {original_size} -> {len(image_bytes)} bytes")

        # Handle different display modes
        if display_mode == "display":
            # Only return ImageContent for inline display
            img_content = self._create_image_content(image_bytes, format="png")
            if img_content:
                return img_content
            else:
                # Fallback if ImageContent creation fails
                logger.warning("Failed to create ImageContent, falling back to metadata")
                display_mode = "save"

        if display_mode == "save":
            # Only save file and return metadata
            file_metadata = self.storage_manager.save_file(
                content=image_bytes,
                file_type="event_study",
                method=method.replace(' ', '_').lower(),
                extension="png",
                custom_path=save_path
            )

            return {
                "status": "success",
                "plot_type": "event_study",
                "method": method,
                "file_path": file_metadata['path'],
                "resource_uri": file_metadata['uri'],
                "file_size": file_metadata['size'],
                "backend": "matplotlib",
                "metadata": file_metadata
            }

        # display_mode == "both": Save file + include display info
        file_metadata = self.storage_manager.save_file(
            content=image_bytes,
            file_type="event_study",
            method=method.replace(' ', '_').lower(),
            extension="png",
            custom_path=save_path
        )

        image_base64 = base64.b64encode(image_bytes).decode()

        result = {
            "status": "success",
            "plot_type": "event_study",
            "method": method,
            "file_path": file_metadata['path'],
            "resource_uri": file_metadata['uri'],
            "file_size": file_metadata['size'],
            "image_base64": image_base64,
            "backend": "matplotlib",
            "metadata": file_metadata,
            "can_display_inline": len(image_bytes) <= max_inline_size,
            "optimized": original_size != len(image_bytes)
        }

        # Add ImageContent if available and size permits
        if len(image_bytes) <= max_inline_size:
            img_content = self._create_image_content(image_bytes, format="png")
            if img_content:
                result["image_content"] = img_content

        return result
    
    def _create_plotly_event_study(
        self,
        event_times: List[int],
        estimates: List[float],
        ci_lower: List[float],
        ci_upper: List[float],
        method: str,
        title: Optional[str],
        save_path: Optional[str],
        show_pretrends: bool,
        confidence_level: float,
        display_mode: str = "both",
        max_inline_size: int = 1_000_000,
        auto_optimize: bool = True
    ) -> Union[Dict[str, Any], ImageContent]:
        """Create event study plot using plotly."""
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=event_times + event_times[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{int(confidence_level*100)}% Confidence Interval',
            showlegend=True
        ))
        
        # Add point estimates
        fig.add_trace(go.Scatter(
            x=event_times,
            y=estimates,
            mode='lines+markers',
            line=dict(color='darkblue', width=3),
            marker=dict(size=8, color='darkblue'),
            name='Point Estimates'
        ))
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="No Effect")
        
        # Add vertical line at treatment time
        fig.add_vline(x=0, line_color="red", 
                     annotation_text="Treatment Start")
        
        # Highlight pre-treatment period
        if show_pretrends:
            pre_treatment_times = [t for t in event_times if t < 0]
            if pre_treatment_times:
                fig.add_vrect(
                    x0=min(pre_treatment_times), x1=0,
                    fillcolor="gray", opacity=0.1,
                    annotation_text="Pre-treatment", annotation_position="top left"
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title or f'Event Study Plot - {method}',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Event Time (Periods Relative to Treatment)',
            yaxis_title='Treatment Effect',
            template='plotly_white',
            width=900,
            height=600,
            legend=dict(x=0.02, y=0.98),
            annotations=[
                dict(
                    text=f'Method: {method}',
                    xref="paper", yref="paper",
                    x=0.02, y=0.02,
                    showarrow=False,
                    bgcolor="wheat",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        # Generate PNG for display
        image_bytes = fig.to_image(format="png", width=900, height=600)

        # Optimize size if needed
        original_size = len(image_bytes)
        if auto_optimize and original_size > max_inline_size:
            image_bytes = self._optimize_image_size(image_bytes, max_inline_size)
            logger.info(f"Plotly image optimized: {original_size} -> {len(image_bytes)} bytes")

        # Handle different display modes
        if display_mode == "display":
            # Only return ImageContent for inline display
            img_content = self._create_image_content(image_bytes, format="png")
            if img_content:
                return img_content
            else:
                logger.warning("Failed to create ImageContent, falling back to metadata")
                display_mode = "save"

        # Convert to HTML for interactive version
        html_content = fig.to_html(include_plotlyjs='cdn')

        if display_mode == "save":
            # Only save files and return metadata
            # Convert save_path to .html if provided
            html_path = None
            if save_path:
                html_path = str(Path(save_path).with_suffix('.html'))
            html_metadata = self.storage_manager.save_file(
                content=html_content,
                file_type="event_study",
                method=method.replace(' ', '_').lower(),
                extension="html",
                custom_path=html_path
            )

            return {
                "status": "success",
                "plot_type": "event_study",
                "method": method,
                "file_path": html_metadata['path'],
                "resource_uri": html_metadata['uri'],
                "file_size": html_metadata['size'],
                "backend": "plotly",
                "metadata": html_metadata
            }

        # display_mode == "both": Save files + include display info
        # Convert save_path to .html if provided
        html_path = None
        if save_path:
            html_path = str(Path(save_path).with_suffix('.html'))
        html_metadata = self.storage_manager.save_file(
            content=html_content,
            file_type="event_study",
            method=method.replace(' ', '_').lower(),
            extension="html",
            custom_path=html_path
        )

        image_base64 = base64.b64encode(image_bytes).decode()

        result = {
            "status": "success",
            "plot_type": "event_study",
            "method": method,
            "file_path": html_metadata['path'],
            "resource_uri": html_metadata['uri'],
            "file_size": html_metadata['size'],
            "image_base64": image_base64,
            "backend": "plotly",
            "metadata": html_metadata,
            "can_display_inline": len(image_bytes) <= max_inline_size,
            "optimized": original_size != len(image_bytes)
        }

        # Add ImageContent if available and size permits
        if len(image_bytes) <= max_inline_size:
            img_content = self._create_image_content(image_bytes, format="png")
            if img_content:
                result["image_content"] = img_content

        return result

    def create_goodman_bacon_plot(
        self,
        bacon_results: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create Goodman-Bacon decomposition plot.

        Args:
            bacon_results: Results from Goodman-Bacon decomposition
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Dict with plot info and base64 encoded image
        """
        if "comparison_types" not in bacon_results:
            return {
                "status": "error",
                "message": "No comparison types data found in Bacon results"
            }

        comparison_types = bacon_results["comparison_types"]
        overall_estimate = bacon_results.get("overall_estimate", 0)

        # Extract data
        comp_names = list(comparison_types.keys())
        weights = [comparison_types[comp]["weight"] for comp in comp_names]
        estimates = [comparison_types[comp]["estimate"] for comp in comp_names]

        if self.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_bacon_plot(
                comp_names, weights, estimates, overall_estimate, title, save_path
            )
        elif self.backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_bacon_plot(
                comp_names, weights, estimates, overall_estimate, title, save_path
            )
        else:
            return {
                "status": "error",
                "message": f"Backend {self.backend} not available"
            }

    def _create_matplotlib_bacon_plot(
        self,
        comp_names: List[str],
        weights: List[float],
        estimates: List[float],
        overall_estimate: float,
        title: Optional[str],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Create Goodman-Bacon plot using matplotlib."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Weights by comparison type
        colors = plt.cm.Set3(np.linspace(0, 1, len(comp_names)))
        bars1 = ax1.bar(range(len(comp_names)), weights, color=colors, alpha=0.8)
        ax1.set_xlabel('Comparison Type', fontsize=12)
        ax1.set_ylabel('Weight', fontsize=12)
        ax1.set_title('Weights by Comparison Type', fontsize=14)
        ax1.set_xticks(range(len(comp_names)))
        ax1.set_xticklabels(comp_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, weight in zip(bars1, weights, strict=False):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=10)

        # Plot 2: Estimates vs Weights (bubble plot)
        sizes = [w * 1000 for w in weights]  # Scale for visibility
        ax2.scatter(estimates, weights, s=sizes, c=colors, alpha=0.6)

        # Add overall estimate line
        ax2.axvline(x=overall_estimate, color='red', linestyle='--',
                   label=f'Overall TWFE: {overall_estimate:.3f}')

        ax2.set_xlabel('Estimate', fontsize=12)
        ax2.set_ylabel('Weight', fontsize=12)
        ax2.set_title('Estimates vs Weights', fontsize=14)
        ax2.legend()

        # Add comparison type labels
        for _, (est, weight, name) in enumerate(zip(estimates, weights, comp_names, strict=False)):
            ax2.annotate(name, (est, weight), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, alpha=0.8)

        plt.suptitle(title or 'Goodman-Bacon Decomposition', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Convert to bytes for saving
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode()

        plt.close()

        # Save using storage manager
        file_metadata = self.storage_manager.save_file(
            content=image_bytes,
            file_type="bacon_decomp",
            extension="png",
            custom_path=save_path
        )

        return {
            "status": "success",
            "plot_type": "goodman_bacon",
            "file_path": file_metadata['path'],
            "resource_uri": file_metadata['uri'],
            "file_size": file_metadata['size'],
            "image_base64": image_base64,
            "backend": "matplotlib",
            "metadata": file_metadata
        }

    def _create_plotly_bacon_plot(
        self,
        comp_names: List[str],
        weights: List[float],
        estimates: List[float],
        overall_estimate: float,
        title: Optional[str],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Create Goodman-Bacon plot using plotly."""

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Weights by Comparison Type', 'Estimates vs Weights'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Plot 1: Bar chart of weights
        fig.add_trace(
            go.Bar(
                x=comp_names,
                y=weights,
                name='Weights',
                text=[f'{w:.3f}' for w in weights],
                textposition='outside',
                marker_color=px.colors.qualitative.Set3[:len(comp_names)]
            ),
            row=1, col=1
        )

        # Plot 2: Scatter plot of estimates vs weights
        fig.add_trace(
            go.Scatter(
                x=estimates,
                y=weights,
                mode='markers+text',
                marker=dict(
                    size=[w * 100 for w in weights],  # Scale for visibility
                    color=px.colors.qualitative.Set3[:len(comp_names)],
                    opacity=0.7
                ),
                text=comp_names,
                textposition='top center',
                name='Comparisons'
            ),
            row=1, col=2
        )

        # Add overall estimate line
        fig.add_vline(
            x=overall_estimate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall TWFE: {overall_estimate:.3f}",
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=title or 'Goodman-Bacon Decomposition',
            title_x=0.5,
            showlegend=False,
            width=1200,
            height=500
        )

        fig.update_xaxes(title_text="Comparison Type", row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_xaxes(title_text="Estimate", row=1, col=2)
        fig.update_yaxes(title_text="Weight", row=1, col=2)

        # Convert to HTML string for saving
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        # Save HTML using storage manager
        # Convert save_path to .html if provided
        html_path = None
        if save_path:
            html_path = str(Path(save_path).with_suffix('.html'))
        html_metadata = self.storage_manager.save_file(
            content=html_content,
            file_type="bacon_decomp",
            extension="html",
            custom_path=html_path
        )

        # Convert to base64 image
        image_bytes = fig.to_image(format="png", width=1200, height=500)
        image_base64 = base64.b64encode(image_bytes).decode()

        return {
            "status": "success",
            "plot_type": "goodman_bacon",
            "file_path": html_metadata['path'],
            "resource_uri": html_metadata['uri'],
            "file_size": html_metadata['size'],
            "image_base64": image_base64,
            "backend": "plotly",
            "metadata": html_metadata
        }

    def create_twfe_weights_plot(
        self,
        weights_results: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create TWFE weights distribution plot.

        Args:
            weights_results: Results from TWFE weights analysis
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Dict with plot info and base64 encoded image
        """
        negative_weight_share = weights_results.get("negative_weight_share", 0)
        n_negative_weights = weights_results.get("n_negative_weights", 0)

        if self.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_weights_plot(
                negative_weight_share, n_negative_weights, title, save_path
            )
        elif self.backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_weights_plot(
                negative_weight_share, n_negative_weights, title, save_path
            )
        else:
            return {
                "status": "error",
                "message": f"Backend {self.backend} not available"
            }

    def _create_matplotlib_weights_plot(
        self,
        negative_weight_share: float,
        n_negative_weights: int,
        title: Optional[str],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Create TWFE weights plot using matplotlib."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Pie chart of weight distribution
        positive_share = 1 - negative_weight_share
        sizes = [positive_share, negative_weight_share]
        labels = ['Positive Weights', 'Negative Weights']
        colors = ['lightgreen', 'lightcoral']
        explode = (0, 0.1)  # Explode negative weights slice

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                          explode=explode, autopct='%1.1f%%',
                                          shadow=True, startangle=90)
        ax1.set_title('TWFE Weight Distribution', fontsize=14)

        # Plot 2: Bar chart showing negative weights concern
        categories = ['Positive\nWeights', 'Negative\nWeights']
        values = [positive_share, negative_weight_share]
        colors_bar = ['green', 'red']

        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7)
        ax2.set_ylabel('Share of Total Weights', fontsize=12)
        ax2.set_title('Weight Distribution Analysis', fontsize=14)
        ax2.set_ylim(0, 1)

        # Add threshold line for concern (5%)
        ax2.axhline(y=0.05, color='orange', linestyle='--',
                   label='Concern Threshold (5%)')
        ax2.legend()

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Add warning text if negative weights are substantial
        if negative_weight_share > 0.05:
            ax2.text(0.5, 0.8, ' Substantial negative\nweights detected!',
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    fontsize=12, fontweight='bold')

        plt.suptitle(title or 'TWFE Weights Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Convert to bytes for saving
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode()

        plt.close()

        # Save using storage manager
        file_metadata = self.storage_manager.save_file(
            content=image_bytes,
            file_type="twfe_weights",
            extension="png",
            custom_path=save_path
        )

        return {
            "status": "success",
            "plot_type": "twfe_weights",
            "file_path": file_metadata['path'],
            "resource_uri": file_metadata['uri'],
            "file_size": file_metadata['size'],
            "image_base64": image_base64,
            "backend": "matplotlib",
            "metadata": file_metadata
        }

    def _create_plotly_weights_plot(
        self,
        negative_weight_share: float,
        n_negative_weights: int,
        title: Optional[str],
        save_path: Optional[str]
    ) -> Dict[str, Any]:
        """Create TWFE weights plot using plotly."""

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Weight Distribution', 'Negative Weights Analysis'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )

        # Plot 1: Pie chart
        positive_share = 1 - negative_weight_share
        fig.add_trace(
            go.Pie(
                labels=['Positive Weights', 'Negative Weights'],
                values=[positive_share, negative_weight_share],
                marker_colors=['lightgreen', 'lightcoral'],
                textinfo='label+percent',
                pull=[0, 0.1]  # Pull out negative weights slice
            ),
            row=1, col=1
        )

        # Plot 2: Bar chart
        fig.add_trace(
            go.Bar(
                x=['Positive<br>Weights', 'Negative<br>Weights'],
                y=[positive_share, negative_weight_share],
                marker_color=['green', 'red'],
                text=[f'{positive_share:.1%}', f'{negative_weight_share:.1%}'],
                textposition='outside',
                name='Weight Share'
            ),
            row=1, col=2
        )

        # Add threshold line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="orange",
            annotation_text="Concern Threshold (5%)",
            row=1, col=2
        )

        # Add warning annotation if needed
        if negative_weight_share > 0.05:
            fig.add_annotation(
                x=0.5, y=0.8,
                text=" Substantial negative<br>weights detected!",
                showarrow=False,
                bgcolor="yellow",
                bordercolor="black",
                borderwidth=1,
                xref="x2", yref="y2"
            )

        # Update layout
        fig.update_layout(
            title_text=title or 'TWFE Weights Analysis',
            title_x=0.5,
            showlegend=False,
            width=1000,
            height=500
        )

        fig.update_yaxes(title_text="Share of Total Weights", range=[0, 1], row=1, col=2)

        # Convert to HTML string for saving
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        # Save HTML using storage manager
        # Convert save_path to .html if provided
        html_path = None
        if save_path:
            html_path = str(Path(save_path).with_suffix('.html'))
        html_metadata = self.storage_manager.save_file(
            content=html_content,
            file_type="twfe_weights",
            extension="html",
            custom_path=html_path
        )

        # Convert to base64 image
        image_bytes = fig.to_image(format="png", width=1000, height=500)
        image_base64 = base64.b64encode(image_bytes).decode()

        return {
            "status": "success",
            "plot_type": "twfe_weights",
            "file_path": html_metadata['path'],
            "resource_uri": html_metadata['uri'],
            "file_size": html_metadata['size'],
            "image_base64": image_base64,
            "backend": "plotly",
            "metadata": html_metadata
        }

    def create_comprehensive_did_report(
        self,
        estimation_results: Dict[str, Any],
        diagnostic_results: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive DID analysis report with multiple plots.

        Args:
            estimation_results: DID estimation results
            diagnostic_results: Optional diagnostic results
            title: Report title
            save_path: Path to save the report

        Returns:
            Dict with all plot info and combined report
        """
        plots = {}

        # Create event study plot
        if "event_study" in estimation_results:
            event_plot = self.create_event_study_plot(
                estimation_results,
                title=f"Event Study - {estimation_results.get('method', 'DID')}"
            )
            if event_plot["status"] == "success":
                plots["event_study"] = event_plot

        # Create diagnostic plots if available
        if diagnostic_results:
            if "bacon_decomp" in diagnostic_results:
                bacon_plot = self.create_goodman_bacon_plot(
                    diagnostic_results["bacon_decomp"],
                    title="Goodman-Bacon Decomposition"
                )
                if bacon_plot["status"] == "success":
                    plots["goodman_bacon"] = bacon_plot

            if "twfe_weights" in diagnostic_results:
                weights_plot = self.create_twfe_weights_plot(
                    diagnostic_results["twfe_weights"],
                    title="TWFE Weights Analysis"
                )
                if weights_plot["status"] == "success":
                    plots["twfe_weights"] = weights_plot

        # Generate HTML report
        html_content = self._generate_html_report(plots, estimation_results, diagnostic_results, title)

        # Save HTML report using storage manager
        report_metadata = self.storage_manager.save_file(
            content=html_content,
            file_type="comprehensive_report",
            method=estimation_results.get('method', 'unknown').replace(' ', '_').lower(),
            extension="html",
            custom_path=save_path
        )

        return {
            "status": "success",
            "report_type": "comprehensive",
            "plots": plots,
            "report_path": report_metadata['path'],
            "resource_uri": report_metadata['uri'],
            "file_size": report_metadata['size'],
            "n_plots": len(plots),
            "metadata": report_metadata
        }

    def _generate_html_report(
        self,
        plots: Dict[str, Any],
        estimation_results: Dict[str, Any],
        diagnostic_results: Optional[Dict[str, Any]],
        title: Optional[str]
    ) -> str:
        """Generate HTML report with embedded plots."""

        def _fmt(val, spec=".4f"):
            """Format numeric values; pass through non-numeric as-is."""
            if isinstance(val, (int, float)):
                return f"{val:{spec}}"
            return str(val)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title or 'DID Analysis Report'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }}
                .success {{ background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title or 'DID Analysis Report'}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        # Add estimation summary
        if estimation_results:
            method = estimation_results.get("method", "Unknown")
            html += f"""
            <div class="section">
                <h2>Estimation Results</h2>
                <div class="summary">
                    <h3>Method: {method}</h3>
            """

            if "overall_att" in estimation_results:
                att = estimation_results["overall_att"]
                html += f"""
                    <p><strong>Overall ATT:</strong> {_fmt(att.get('estimate', 'N/A'))}</p>
                    <p><strong>Standard Error:</strong> {_fmt(att.get('se', 'N/A'))}</p>
                    <p><strong>95% CI:</strong> [{_fmt(att.get('ci_lower', 'N/A'))}, {_fmt(att.get('ci_upper', 'N/A'))}]</p>
                    <p><strong>P-value:</strong> {_fmt(att.get('pvalue', 'N/A'))}</p>
                """

            html += "</div></div>"

        # Add diagnostic summary
        if diagnostic_results:
            html += """
            <div class="section">
                <h2>Diagnostic Results</h2>
            """

            # Goodman-Bacon summary
            if "bacon_decomp" in diagnostic_results:
                bacon = diagnostic_results["bacon_decomp"]
                forbidden_weight = bacon.get("forbidden_comparison_weight", 0)

                if forbidden_weight > 0.1:
                    html += f"""
                    <div class="warning">
                        <h4> TWFE Bias Warning</h4>
                        <p>High weight on forbidden comparisons: {forbidden_weight:.1%}</p>
                        <p>TWFE estimates may be biased. Consider using robust estimators.</p>
                    </div>
                    """
                else:
                    html += f"""
                    <div class="success">
                        <h4>TWFE Appears Unproblematic</h4>
                        <p>Low weight on forbidden comparisons: {forbidden_weight:.1%}</p>
                    </div>
                    """

            # TWFE weights summary
            if "twfe_weights" in diagnostic_results:
                weights = diagnostic_results["twfe_weights"]
                neg_weight_share = weights.get("negative_weight_share", 0)

                if neg_weight_share > 0.05:
                    html += f"""
                    <div class="warning">
                        <h4> Negative Weights Warning</h4>
                        <p>Substantial negative weights: {neg_weight_share:.1%}</p>
                        <p>Consider using heterogeneity-robust estimators.</p>
                    </div>
                    """

            html += "</div>"

        # Add plots
        for plot_type, plot_info in plots.items():
            if plot_info["status"] == "success":
                html += f"""
                <div class="section">
                    <h2>{plot_type.replace('_', ' ').title()}</h2>
                    <div class="plot">
                        <img src="data:image/png;base64,{plot_info['image_base64']}" alt="{plot_type}">
                    </div>
                </div>
                """

        html += """
        </body>
        </html>
        """

        return html

    def _optimize_image_size(self, img_bytes: bytes, max_size: int = 1_000_000) -> bytes:
        """
        Optimize image size to meet display constraints.

        Implements progressive optimization strategies:
        1. Lower DPI if needed
        2. Resize image dimensions
        3. Convert to JPEG with compression

        Args:
            img_bytes: Original image bytes
            max_size: Maximum allowed size in bytes (default: 1MB)

        Returns:
            Optimized image bytes
        """
        if len(img_bytes) <= max_size:
            return img_bytes

        logger.info(f"Image size {len(img_bytes)} bytes exceeds {max_size} bytes, optimizing...")

        try:
            from PIL import Image as PILImage

            # Load image
            img = PILImage.open(io.BytesIO(img_bytes))

            # Strategy 1: Lower DPI progressively
            for dpi in [150, 100, 72]:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', dpi=(dpi, dpi), optimize=True)
                optimized_size = buffer.tell()
                logger.debug(f"DPI {dpi}: size = {optimized_size} bytes")

                if optimized_size <= max_size:
                    logger.info(f"Optimized to {optimized_size} bytes by lowering DPI to {dpi}")
                    return buffer.getvalue()

            # Strategy 2: Resize dimensions progressively
            scale = 0.8
            while scale > 0.3:
                new_size = tuple(int(dim * scale) for dim in img.size)
                resized = img.resize(new_size, PILImage.Resampling.LANCZOS)
                buffer = io.BytesIO()
                resized.save(buffer, format='PNG', optimize=True)
                optimized_size = buffer.tell()
                logger.debug(f"Scale {scale:.1f}: size = {optimized_size} bytes")

                if optimized_size <= max_size:
                    logger.info(f"Optimized to {optimized_size} bytes by resizing to {new_size}")
                    return buffer.getvalue()
                scale -= 0.1

            # Strategy 3: Convert to JPEG with compression
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            for quality in [85, 75, 65, 50]:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                optimized_size = buffer.tell()
                logger.debug(f"JPEG quality {quality}: size = {optimized_size} bytes")

                if optimized_size <= max_size:
                    logger.info(f"Optimized to {optimized_size} bytes using JPEG quality {quality}")
                    return buffer.getvalue()

            # If still too large, return best effort
            logger.warning(f"Could not optimize image below {max_size} bytes")
            return buffer.getvalue()

        except Exception as e:
            logger.exception("Error optimizing image")
            return img_bytes

    def _create_image_content(self, img_bytes: bytes, format: str = "png") -> Optional[ImageContent]:
        """
        Create ImageContent object for Claude Desktop display.

        Args:
            img_bytes: Image data as bytes
            format: Image format (png, jpeg, etc.)

        Returns:
            ImageContent object or None if not available
        """
        if not FASTMCP_IMAGE_AVAILABLE:
            logger.warning("FastMCP image types not available")
            return None

        try:
            img_obj = Image(data=img_bytes, format=format)
            return img_obj.to_image_content()
        except Exception as e:
            logger.exception("Error creating ImageContent")
            return None

    def get_available_backends(self) -> List[str]:
        """Get list of available visualization backends."""
        backends = []
        if MATPLOTLIB_AVAILABLE:
            backends.append("matplotlib")
        if PLOTLY_AVAILABLE:
            backends.append("plotly")
        return backends

    def set_backend(self, backend: str) -> bool:
        """
        Set visualization backend.

        Args:
            backend: "matplotlib" or "plotly"

        Returns:
            True if successful, False otherwise
        """
        available = self.get_available_backends()
        if backend in available:
            self.backend = backend
            logger.info(f"Visualization backend set to {backend}")
            return True
        else:
            logger.error(f"Backend {backend} not available. Available: {available}")
            return False
