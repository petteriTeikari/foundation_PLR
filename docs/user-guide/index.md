# User Guide

This guide explains the Foundation PLR pipeline architecture and how to use each component.

## Pipeline Overview

The Foundation PLR pipeline consists of four main stages:

```mermaid
graph TB
    subgraph "Stage 1: Outlier Detection"
        OD[11 Methods]
        OD1[pupil-gt<br/>Ground Truth]
        OD2[MOMENT/UniTS/TimesNet<br/>Foundation Models]
        OD3[LOF/SVM/PROPHET<br/>Traditional]
    end

    subgraph "Stage 2: Imputation"
        IMP[8 Methods]
        IMP1[pupil-gt<br/>Ground Truth]
        IMP2[SAITS/CSDI<br/>Deep Learning]
        IMP3[MOMENT<br/>Foundation Model]
    end

    subgraph "Stage 3: Featurization"
        FEAT[Handcrafted Features]
        FEAT1[Amplitude Bins]
        FEAT2[Latency Features]
    end

    subgraph "Stage 4: Classification"
        CLS[CatBoost<br/>Fixed Classifier]
    end

    OD --> IMP --> FEAT --> CLS
```

## Sections

### Pipeline Stages

- **[Pipeline Overview](pipeline-overview.md)** - Detailed architecture explanation
- **[Outlier Detection](outlier-detection.md)** - Available methods and configuration
- **[Imputation](imputation.md)** - Signal reconstruction approaches
- **[Featurization](featurization.md)** - Feature extraction from PLR signals
- **[Classification](classification.md)** - Model training and evaluation

### Infrastructure

- **[Prefect Orchestration](prefect-orchestration.md)** - Workflow orchestration with Prefect
