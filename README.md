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

   **⚠️ CRITICAL for Claude Desktop users on macOS:**

   You **MUST** install uv via Homebrew (not curl) to ensure Claude Desktop can find it:

   ```bash
   # Required method for Claude Desktop
   brew install uv
   ```

   **Why Homebrew?** Claude Desktop runs in an isolated environment and can only find `uv` if it's installed via Homebrew (which places it in `/opt/homebrew/bin/` or `/usr/local/bin/`).

   **Already installed uv with curl?** Create a symlink to make it accessible:
   ```bash
   # If you installed with: curl -LsSf https://astral.sh/uv/install.sh | sh
   # Run this to fix it:
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

   Verify installation:
   ```bash
   uv --version  # Should display version number
   which uv      # Should show /opt/homebrew/bin/uv or /usr/local/bin/uv
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

# IMPORTANT: Verify uv is installed correctly FIRST
which uv  # Should show /opt/homebrew/bin/uv or /usr/local/bin/uv
# If not, run: brew install uv

# Install with FastMCP (automatically configures Claude Desktop)
pip install fastmcp  # Install fastmcp first
fastmcp install claude-desktop src/chatdid_mcp/server.py --name "ChatDiD" --project $(pwd)
```

Then restart Claude Desktop completely (quit and reopen). Look for the hammer icon (🔨) to confirm the server is loaded.

**If you see "Server disconnected" error**, see the [Troubleshooting](#troubleshooting-installation) section below.

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
       "ChatDiD": {
         "command": "/ABSOLUTE/PATH/TO/uv",
         "args": [
           "run",
           "--project",
           "/ABSOLUTE/PATH/TO/ChatDiD",
           "--with",
           "fastmcp",
           "fastmcp",
           "run",
           "/ABSOLUTE/PATH/TO/ChatDiD/src/chatdid_mcp/server.py"
         ]
       }
     }
   }
   ```

   **Important Notes:**
   - Replace `/ABSOLUTE/PATH/TO/ChatDiD` with your actual ChatDiD path (e.g., `/Users/yourname/ChatDiD`)
   - Replace `/ABSOLUTE/PATH/TO/uv` with your uv installation path:
     - Run `which uv` to find the path (commonly `/Users/yourname/.local/bin/uv` on macOS)
     - Or use full Homebrew path: `/opt/homebrew/bin/uv` (Apple Silicon) or `/usr/local/bin/uv` (Intel Mac)
   - All paths must be absolute, not relative

   **For VSCode with GitHub Copilot:**

   Create `.vscode/mcp.json` in your workspace:
   ```json
   {
     "mcpServers": {
       "chatdid": {
         "command": "uv",
         "args": [
           "run",
           "--project",
           "/ABSOLUTE/PATH/TO/ChatDiD",
           "--with",
           "fastmcp",
           "fastmcp",
           "run",
           "/ABSOLUTE/PATH/TO/ChatDiD/src/chatdid_mcp/server.py"
         ]
       }
     }
   }
   ```

   **For Cline (VSCode extension):**

   Open Cline → MCP Servers icon → Advanced MCP Settings, then add the same configuration above.

   **For Claude Code:**

   You can configure ChatDiD MCP server using either the CLI or by editing the configuration file:

   **Method 1: Using CLI (Recommended)**
   ```bash
   # Add ChatDiD with user scope (available in all projects)
   claude mcp add --transport stdio --scope user ChatDiD -- \
     uv run --project /ABSOLUTE/PATH/TO/ChatDiD --with fastmcp \
     fastmcp run /ABSOLUTE/PATH/TO/ChatDiD/src/chatdid_mcp/server.py

   # Verify installation
   claude mcp list
   ```

   **Method 2: Direct Configuration File Edit**

   Edit `~/.claude.json` (create if it doesn't exist):
   ```json
   {
     "mcpServers": {
       "ChatDiD": {
         "command": "uv",
         "args": [
           "run",
           "--project",
           "/ABSOLUTE/PATH/TO/ChatDiD",
           "--with",
           "fastmcp",
           "fastmcp",
           "run",
           "/ABSOLUTE/PATH/TO/ChatDiD/src/chatdid_mcp/server.py"
         ]
       }
     }
   }
   ```

   **Important Notes for Claude Code:**
   - Replace `/ABSOLUTE/PATH/TO/ChatDiD` with your actual ChatDiD directory path
   - Use `--scope user` for personal tools available across all projects
   - Use `--scope project` to share with your team (creates `.mcp.json` in project root)
   - After configuration, restart Claude Code or run `claude mcp list` to verify

3. Restart your MCP client completely

### Troubleshooting Installation

#### Error: "spawn uv ENOENT" or "Server disconnected"

This is the most common installation error. It means Claude Desktop cannot find the `uv` command.

**Quick Fix:**

1. Check your uv installation:
   ```bash
   which uv
   ```

2. If it shows `~/.local/bin/uv`:
   ```bash
   # Create symlink to make it accessible to Claude Desktop
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

3. If uv is not installed, install it with Homebrew:
   ```bash
   brew install uv
   ```

4. Restart Claude Desktop completely (Cmd+Q to quit, then reopen)

**Still not working?** Check the logs:
```bash
# Open logs folder
open ~/Library/Logs/Claude/

# Look for mcp-server-ChatDiD.log and check for errors
tail -50 ~/Library/Logs/Claude/mcp-server-ChatDiD.log
```

See [docs/FASTMCP_TROUBLESHOOTING.md](docs/FASTMCP_TROUBLESHOOTING.md) for detailed troubleshooting guide.

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

### Core DID Estimators

- **Callaway & Sant'Anna (2021)**: Group-time average treatment effects with doubly robust estimation
- **Sun & Abraham (2021)**: Interaction-weighted estimator using fixest
- **Borusyak, Jaravel & Spiess (2024)**: Imputation-based estimator
- **Gardner (2022)**: Two-stage difference-in-differences
- **de Chaisemartin & D'Haultfoeuille (2020)**: Multiple treatment periods with heterogeneous effects
- **Roth & Sant'Anna (2023)**: Efficient GMM estimator

### Synthetic Control Methods

- **gsynth (Xu 2017)**: Generalized synthetic control with interactive fixed effects
- **synthdid (Arkhangelsky et al. 2019)**: Synthetic DiD combining SC and DiD

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
