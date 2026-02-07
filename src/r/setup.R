# Foundation PLR R Environment Setup
# Run this script once to install all required packages
# Created: 2026-01-25

# ==============================================================================
# CRAN PACKAGES
# ==============================================================================

cran_packages <- c(
  # Core ggplot2 ecosystem
  "ggplot2",
  "ggdist",           # Half-eye, raincloud plots
  "ggbeeswarm",       # Better beeswarm plots
  "ggalluvial",       # Sankey/alluvial diagrams
  "ggExtra",          # Marginal histograms
  "patchwork",        # Multi-panel layouts
  "cowplot",          # Publication-ready themes

  # Data manipulation
  "dplyr",
  "tidyr",
  "readr",
  "jsonlite",

  # Visualization utilities
  "viridis",          # Colorblind-safe continuous scales
  "scales",           # Scale functions
  "ragg",             # High-quality graphics device

  # Statistical visualization
  "dotwhisker",       # Coefficient plots
  "scmamp",           # Critical difference diagrams

  # Environment management
  "renv"
)

# Check and install missing packages
install_if_missing <- function(packages) {
  missing <- packages[!packages %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    message("Installing: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  } else {
    message("All packages already installed.")
  }
}

install_if_missing(cran_packages)

# ==============================================================================
# OPTIONAL: COLORBLIND TESTING
# ==============================================================================

# colorblindr requires cowplot and colorspace
if (!requireNamespace("colorblindr", quietly = TRUE)) {
  message("Installing colorblindr for accessibility testing...")
  install.packages("colorspace")
  # colorblindr is on GitHub only
  if (requireNamespace("remotes", quietly = TRUE)) {
    remotes::install_github("clauswilke/colorblindr")
  } else {
    install.packages("remotes")
    remotes::install_github("clauswilke/colorblindr")
  }
}

# ==============================================================================
# LOAD FOUNDATION PLR MODULES
# ==============================================================================

# Source theme and color palettes
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir)) script_dir <- "r"

source(file.path(script_dir, "theme_foundation_plr.R"))
source(file.path(script_dir, "color_palettes.R"))
source(file.path(script_dir, "load_data.R"))

message("Foundation PLR R environment ready.")
message("Use theme_foundation_plr() and scale_color_pipeline() for figures.")
