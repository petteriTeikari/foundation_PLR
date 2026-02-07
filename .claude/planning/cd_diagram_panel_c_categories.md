# Plan: CD Diagram Panel C - 5 Category Comparison

## Current State
- Panel A: 11 individual outlier detection methods (CORRECT)
- Panel B: 5 individual imputation methods (CORRECT)
- Panel C: 8 individual combined pipelines (WRONG - should show 5 categories)

## Target State
- Panel C: 5 preprocessing categories (same as other figures):
  1. Ground Truth
  2. Ensemble FM
  3. Single-model FM
  4. Deep Learning
  5. Traditional

## Category Assignment (by outlier method)

From `configs/mlflow_registry/category_mapping.yaml`:

| Outlier Method | Raw Category | Display Category |
|----------------|--------------|------------------|
| pupil-gt | Ground Truth | Ground Truth |
| ensemble-* | Ensemble | Ensemble FM |
| MOMENT-*, UniTS-* | Foundation Model | Single-model FM |
| TimesNet-* | Deep Learning | Deep Learning |
| LOF, OneClassSVM, PROPHET, SubPCA | Traditional | Traditional |

## Implementation Approach

1. For Panel C, aggregate AUROC by preprocessing category
2. Use imputation methods as "folds" (replicates) within each category
3. Each category will have N replicates where N = # imputation methods

## TDD Test Cases

### Test 1: Category assignment from method names
```r
test_that("outlier methods map to correct categories", {
  expect_equal(get_outlier_category("pupil-gt"), "Ground Truth")
  expect_equal(get_outlier_category("MOMENT-gt-finetune"), "Foundation Model")
  expect_equal(get_outlier_category("ensemble-LOF-MOMENT-..."), "Ensemble")
})
```

### Test 2: Display name mapping
```r
test_that("categories map to display names", {
  expect_equal(to_display_category("Foundation Model"), "Single-model FM")
  expect_equal(to_display_category("Ensemble"), "Ensemble FM")
})
```

### Test 3: Category aggregation produces 5 groups
```r
test_that("aggregation produces exactly 5 categories", {
  df <- read_metrics()
  agg <- aggregate_by_category(df)
  expect_equal(length(unique(agg$category)), 5)
})
```

### Test 4: CD matrix has correct structure for categories
```r
test_that("category CD matrix has 5 columns", {
  mat <- create_category_cd_matrix(df)
  expect_equal(ncol(mat), 5)
})
```

## Files to Modify

1. `src/r/figures/fig_cd_diagrams.R` - Add category aggregation for Panel C
2. `src/r/figure_system/category_loader.R` - Add display name mapping function
3. `tests/r/test_cd_diagram_categories.R` - New test file

## Category Order (consistent with other figures)

```r
category_order <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
```
