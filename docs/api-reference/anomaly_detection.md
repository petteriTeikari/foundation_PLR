# anomaly_detection

Outlier detection methods for PLR signals.

## Overview

This module provides 15 different outlier detection methods:

- **Ground Truth**: Human-annotated masks
- **Foundation Models**: MOMENT, UniTS, TimesNet
- **Traditional**: LOF, OneClassSVM, PROPHET
- **Ensembles**: Voting combinations

## Main Entry Point

::: src.anomaly_detection.flow_anomaly_detection
    options:
      show_root_heading: true
      members_order: source

## Core Functions

::: src.anomaly_detection.anomaly_detection
    options:
      show_root_heading: true
      members_order: source

::: src.anomaly_detection.anomaly_utils
    options:
      show_root_heading: true
      members_order: source

## Traditional Methods

::: src.anomaly_detection.outlier_sklearn
    options:
      show_root_heading: true
      members_order: source

::: src.anomaly_detection.outlier_prophet
    options:
      show_root_heading: true
      members_order: source

## TimesNet Integration

::: src.anomaly_detection.timesnet_wrapper
    options:
      show_root_heading: true
      members_order: source

## Logging

::: src.anomaly_detection.log_anomaly_detection
    options:
      show_root_heading: true
      members_order: source
