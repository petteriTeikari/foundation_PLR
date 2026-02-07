# classification

Model training and evaluation for glaucoma classification.

## Overview

This module handles:

- Bootstrap evaluation with 1000 iterations
- STRATOS-compliant metric computation
- Multiple classifier support (CatBoost default)
- Subject-wise analysis

## Main Entry Point

::: src.classification.flow_classification
    options:
      show_root_heading: true
      members_order: source

## Bootstrap Evaluation

::: src.classification.bootstrap_evaluation
    options:
      show_root_heading: true
      members_order: source

## Metrics and Statistics

::: src.classification.stats_metric_utils
    options:
      show_root_heading: true
      members_order: source

## Classifier Utilities

::: src.classification.classifier_utils
    options:
      show_root_heading: true
      members_order: source

::: src.classification.classifier_evaluation
    options:
      show_root_heading: true
      members_order: source

## Other Classifiers

::: src.classification.sklearn_simple_classifiers
    options:
      show_root_heading: true
      members_order: source

::: src.classification.tabpfn_main
    options:
      show_root_heading: true
      members_order: source
