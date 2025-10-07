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

1. **Python 3.9+** with the following packages:
   ```bash
   pip install fastmcp pandas numpy scipy statsmodels rpy2 matplotlib seaborn plotly
   ```

2. **R** with DID packages:
   ```r
   install.packages(c("did", "bacondecomp", "fixest", "HonestDiD", "pretrends"))
   ```

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ChatDiD
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

#### Option 1: Direct execution
```bash
python run_server.py
```

#### Option 2: Using FastMCP CLI
```bash
fastmcp run src/chatdid_mcp/server.py
```

### Connecting to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chatdid": {
      "command": "python",
      "args": ["/path/to/ChatDiD/run_server.py"]
    }
  }
}
```

## Usage Example

Once connected to Claude Desktop, you can start a DID analysis:

```
User: "I want to analyze the effect of a policy change using difference-in-differences. 
       I have panel data in a CSV file."

Claude: I'll help you conduct a robust DID analysis! Let's start by loading your data.

[Claude uses the load_data tool]

User: "The file is at /path/to/my/data.csv"

Claude: [Loads data and provides summary]
        Now let's explore the panel structure to identify your variables.

[Claude uses explore_data tool and continues the analysis...]
```

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
  url={https://github.com/chatdid/chatdid-mcp-server}
}
```

## Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: See the built-in guides and resources
- **Community**: Join our discussions for help and tips
