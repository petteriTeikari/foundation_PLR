# featurization

Feature extraction from PLR signals.

## Overview

This module extracts handcrafted physiological features:

- Amplitude bins (histogram features)
- Latency features (timing)
- Velocity features
- PIPR (Post-Illumination Pupil Response)

## Main Entry Point

::: src.featurization.flow_featurization

## PLR Featurization

::: src.featurization.featurize_PLR

::: src.featurization.featurizer_PLR_subject

## Feature Utilities

::: src.featurization.feature_utils

::: src.featurization.feature_log

## Handcrafted Features

::: src.featurization.subflow_handcrafted_featurization

## Embedding Features

::: src.featurization.embedding.subflow_embedding

::: src.featurization.embedding.moment_embedding

::: src.featurization.embedding.dim_reduction

## Visualization

::: src.featurization.visualize_features
