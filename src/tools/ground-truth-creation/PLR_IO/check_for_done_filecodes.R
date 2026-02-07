# check_for_done_filecodes.R
#
# Legacy utility for tracking which PLR recordings have been annotated.
# This file is preserved for documentation purposes.
#
# Original function: Scans output directories for completed annotation files
# and returns a vector of filecodes that have already been processed.
#
# Used by: inspect_outliers, inspect_EMD Shiny apps
#
# Note: This is a placeholder stub. The original implementation read file
# listing from the annotation output directories to skip already-completed files.

check_for_done_filecodes <- function(output_dir) {
  # Placeholder - returns empty vector
  # Original: listed files in output_dir and extracted filecodes
  warning("check_for_done_filecodes is a placeholder stub - returning empty vector")
  return(character(0))
}
