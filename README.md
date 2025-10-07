# ChatDiD MCP Server

A Model Context Protocol (MCP) server for interactive Difference-in-Differences analysis through chat. This server enables AI assistants like Claude to perform robust DID analysis using modern econometric methods.

## Features

- **Interactive DID Analysis**: Conduct complete DID analysis through natural language conversations
- **Modern Methods**: Implements state-of-the-art heterogeneity-robust estimators
- **R Integration**: Leverages powerful R packages for econometric analysis
- **Diagnostic Tools**: Built-in TWFE bias detection and sensitivity analysis
- **User-Friendly**: Designed for both economists and non-experts

## Quick Start

### Prerequisites

1. **Python 3.10+** - Required for MCP SDK

2. **uv package manager** - Required for dependency management:
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or via Homebrew (recommended for macOS)
   brew install uv
   ```

3. **R (4.0+)** with DID packages:
   ```bash
   # Run the installation script
   Rscript install_r_packages.R
   ```

### Installation

#### Option 1: One-Click Install (Recommended)

The easiest way to install ChatDiD in Claude Desktop:

```bash
# Clone the repository
git clone https://github.com/cafferychen777/ChatDiD.git
cd ChatDiD

# Install with FastMCP (automatically configures Claude Desktop)
fastmcp install claude-desktop src/chatdid_mcp/server.py --server-name "ChatDiD"
```

Then restart Claude Desktop completely (quit and reopen).

#### Option 2: Manual Installation

For more control or to use with other MCP clients (Cline, VSCode):

1. Clone and set up the project:
   ```bash
   git clone https://github.com/cafferychen777/ChatDiD.git
   cd ChatDiD

   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv pip install -r requirements.txt
   ```

2. Configure your MCP client:

   **For Claude Desktop:**

   Edit the configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

   Add:
   ```json
   {
     "mcpServers": {
       "chatdid": {
         "command": "uv",
         "args": [
           "--directory",
           "/ABSOLUTE/PATH/TO/ChatDiD",
           "run",
           "fastmcp",
           "run",
           "src/chatdid_mcp/server.py"
         ]
       }
     }
   }
   ```

   **For VSCode with GitHub Copilot:**

   Create `.vscode/mcp.json` in your workspace:
   ```json
   {
     "mcpServers": {
       "chatdid": {
         "command": "uv",
         "args": [
           "--directory",
           "/ABSOLUTE/PATH/TO/ChatDiD",
           "run",
           "fastmcp",
           "run",
           "src/chatdid_mcp/server.py"
         ]
       }
     }
   }
   ```

   **For Cline (VSCode extension):**

   Open Cline → MCP Servers → Advanced MCP Settings, then add the same configuration.

3. Restart your MCP client completely

### Verifying Installation

Once installed, open your MCP client and check that the ChatDiD tools are available:
- `load_data`
- `explore_data`
- `diagnose_twfe`
- `estimate_*` (various estimators)

Try loading example data:
```
Load the mpdta dataset from data/examples/mpdta.csv
```

## Usage Example

Once connected to Claude Desktop (or other MCP clients), you can start a DID analysis through natural conversation:

```
User: "I want to analyze minimum wage policy effects using the mpdta dataset.
       Can you help me run a difference-in-differences analysis?"

Claude: I'll help you conduct a robust DID analysis! Let me start by loading the data.

[Uses load_data tool]

✅ Data Loading Successful
- 2,500 observations × 6 columns
- Columns: year, countyreal, lpop, lemp, first.treat, treat

Let me explore the panel structure to understand your data better.

[Uses explore_data tool]

📊 Panel Structure:
- Unit variable: countyreal
- Time variable: year
- Treatment: 38.2% treated
- Suggests staggered adoption design

Now let's check if standard TWFE would be problematic here.

[Uses diagnose_twfe tool]

⚠️ TWFE Diagnostic Results:
- Goodman-Bacon decomposition shows 13.7% problematic comparisons
- TWFE estimate: -0.0365
- Recommendation: Use heterogeneity-robust estimators

I'll use the Callaway & Sant'Anna estimator, which is robust to these issues.

[Uses estimate_callaway_santanna]

📈 Results:
- Overall ATT: -0.0305 (SE: 0.0129)
- 95% CI: [-0.0558, -0.0051]
- Statistically significant (p = 0.0184)

The treatment shows a significant negative effect on log employment.
Let me create an event study plot to visualize the dynamics.

[Uses create_event_study_plot]

✅ Event study plot saved! The plot shows:
- No pre-treatment trends (parallel trends assumption holds)
- Treatment effects emerge after period 0
- Effects grow stronger over time

User: "Can you export the results?"

Claude: Absolutely! I'll export both the estimates and comparison with other methods.

[Uses export_results and export_comparison]

✅ Results exported to CSV, Excel, and LaTeX formats!
```

This interactive workflow demonstrates how ChatDiD guides you through:
1. **Data loading** with automatic structure detection
2. **Diagnostics** to identify TWFE bias
3. **Method selection** based on data characteristics
4. **Robust estimation** with heterogeneity-robust estimators
5. **Visualization** with publication-ready plots
6. **Export** in multiple formats

## Available Tools

- **`load_data`**: Load datasets from various formats (CSV, Excel, Stata, Parquet)
- **`explore_data`**: Analyze panel structure and identify DID variables
- **`diagnose_twfe`**: Check for TWFE bias using modern diagnostics
- **`estimate_did`**: Estimate treatment effects using robust methods
- **`visualize_results`**: Create event study plots and visualizations
- **`sensitivity_analysis`**: Test robustness of findings
- **`export_results`**: Export analysis results and reports

## Supported Methods

- **Callaway & Sant'Anna (2021)**: Group-time average treatment effects
- **Sun & Abraham (2021)**: Interaction-weighted estimator
- **Borusyak, Jaravel & Spiess (2021)**: Imputation estimator
- **Gardner (2022)**: Two-stage estimator
- **de Chaisemartin & D'Haultfoeuille**: For complex treatments

## Resources

The server provides built-in guides and documentation:

- **Analysis Workflow Guide**: Step-by-step DID analysis process
- **Method Comparison**: When to use which estimator
- **Diagnostic Results**: Real-time analysis results

## Development

### Project Structure

```
ChatDiD/
├── src/chatdid_mcp/
│   ├── __init__.py
│   ├── server.py          # Main MCP server
│   ├── did_analyzer.py    # Core DID analysis engine
│   └── tools/             # Individual MCP tools
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use ChatDiD in your research, please cite:

```bibtex
@software{chatdid2024,
  title={ChatDiD: Interactive Difference-in-Differences Analysis through Chat},
  author={ChatDiD Team},
  year={2024},
  url={https://github.com/cafferychen777/ChatDiD}
}
```

## Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: See the built-in guides and resources
- **Community**: Join our discussions for help and tips
