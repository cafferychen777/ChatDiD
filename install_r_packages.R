#!/usr/bin/env Rscript

# Install all required R packages for ChatDiD MCP Server
# This script installs packages needed for heterogeneity-robust DID estimation

cat("========================================\n")
cat("ChatDiD R Package Installation\n")
cat("========================================\n\n")

# Check if package is installed
check_and_install <- function(pkg, description = "") {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s", pkg))
    if (description != "") {
      cat(sprintf(" (%s)", description))
    }
    cat("...\n")

    tryCatch({
      install.packages(pkg, repos = "https://cloud.r-project.org/")
      cat(sprintf("[OK] %s installed successfully\n\n", pkg))
    }, error = function(e) {
      cat(sprintf("[ERROR] Failed to install %s: %s\n\n", pkg, e$message))
    })
  } else {
    cat(sprintf("[OK] %s is already installed\n", pkg))
  }
}

# Core DID estimators (CRITICAL - required for basic functionality)
cat("== Core DID Estimators ==\n")
check_and_install("did", "Callaway & Sant'Anna (2021)")
check_and_install("fixest", "Sun & Abraham (2021)")
check_and_install("didimputation", "Borusyak, Jaravel & Spiess (2024)")
check_and_install("did2s", "Gardner (2022)")
check_and_install("DIDmultiplegt", "de Chaisemartin & D'Haultfoeuille (2020)")
check_and_install("DIDmultiplegtDYN", "de Chaisemartin & D'Haultfoeuille (2020) - Modern")
check_and_install("staggered", "Roth & Sant'Anna (2023) Efficient Estimator")

cat("\n== Diagnostic Tools ==\n")
check_and_install("bacondecomp", "Goodman-Bacon Decomposition")
check_and_install("TwoWayFEWeights", "TWFE Weights Analysis")

cat("\n== Robustness & Sensitivity ==\n")
check_and_install("HonestDiD", "Rambachan & Roth (2023) Sensitivity Analysis")
check_and_install("pretrends", "Roth (2022) Power Analysis")

cat("\n== Synthetic Control Methods ==\n")
check_and_install("gsynth", "Xu (2017) Generalized Synthetic Control")

# synthdid requires devtools for GitHub installation
if (!requireNamespace("devtools", quietly = TRUE)) {
  cat("Installing devtools (required for synthdid)...\n")
  install.packages("devtools", repos = "https://cloud.r-project.org/")
}

cat("Installing synthdid (Arkhangelsky et al. 2019)...\n")
tryCatch({
  devtools::install_github("synth-inference/synthdid", quiet = TRUE)
  cat("[OK] synthdid installed successfully\n\n")
}, error = function(e) {
  cat(sprintf("[ERROR] Failed to install synthdid: %s\n", e$message))
  cat("Note: synthdid is optional. Install manually with:\n")
  cat("  devtools::install_github('synth-inference/synthdid')\n\n")
})

cat("\n========================================\n")
cat("Installation Summary\n")
cat("========================================\n")

# Verify all critical packages
critical_packages <- c(
  "did", "fixest", "didimputation", "did2s",
  "DIDmultiplegt", "DIDmultiplegtDYN", "staggered",
  "bacondecomp", "TwoWayFEWeights",
  "HonestDiD", "pretrends",
  "gsynth", "synthdid"
)

installed <- sapply(critical_packages, function(pkg) {
  requireNamespace(pkg, quietly = TRUE)
})

cat(sprintf("\nCritical packages installed: %d/%d\n", sum(installed), length(critical_packages)))

if (all(installed)) {
  cat("\n[OK] All critical packages successfully installed!\n")
  cat("[OK] ChatDiD is ready to use.\n\n")
} else {
  cat("\n[ERROR] Some critical packages failed to install:\n")
  missing <- critical_packages[!installed]
  for (pkg in missing) {
    cat(sprintf("  - %s\n", pkg))
  }
  cat("\nPlease install missing packages manually:\n")
  cat(sprintf('install.packages(c("%s"))\n\n', paste(missing, collapse = '", "')))
}

cat("\n========================================\n")
cat("Package Descriptions\n")
cat("========================================\n")
cat("
Core Estimators:
  • did              - Callaway & Sant'Anna doubly robust estimator
  • fixest           - Sun & Abraham interaction-weighted estimator
  • didimputation    - Borusyak, Jaravel & Spiess imputation estimator
  • did2s            - Gardner two-stage estimator
  • DIDmultiplegt    - de Chaisemartin & D'Haultfoeuille estimator (legacy)
  • DIDmultiplegtDYN - de Chaisemartin & D'Haultfoeuille estimator (modern)
  • staggered        - Roth & Sant'Anna efficient estimator

Diagnostics:
  • bacondecomp      - Goodman-Bacon (2021) decomposition
  • TwoWayFEWeights  - de Chaisemartin & D'Haultfoeuille (2020) weights

Robustness:
  • HonestDiD        - Rambachan & Roth (2023) sensitivity analysis
  • pretrends        - Roth (2022) pre-trends power analysis

Synthetic Control:
  • gsynth           - Xu (2017) generalized synthetic control
  • synthdid         - Arkhangelsky et al. (2019) synthetic DiD
\n")

cat("For more information, see:\n")
cat("  https://github.com/cafferychen777/ChatDiD\n\n")
