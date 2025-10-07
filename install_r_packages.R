#!/usr/bin/env Rscript

# Install missing R packages for ChatDiD

# Check if package is installed
check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  } else {
    cat(paste(pkg, "is already installed\n"))
  }
}

# Install DIDmultiplegt
check_package("DIDmultiplegt")

# Install did2s
check_package("did2s")

# Install didimputation
check_package("didimputation")

cat("\nAll packages checked/installed!\n")