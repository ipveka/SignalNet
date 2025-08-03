# Time Series Forecasting Model Improvements Requirements

## Introduction

The current SignalNet model violates fundamental time series forecasting best practices, resulting in meaningless predictions (values ~0 vs ground truth ~1-2). The model lacks proper normalization, appropriate architecture for sequential data, and essential time series features. This specification addresses these critical issues following established time series neural network practices.

## Requirements

### Requirement 1: Proper Time Series Architecture

**User Story:** As a data scientist, I want a model architecture specifically designed for time series forecasting, so that it can learn temporal dependencies and patterns effectively.

#### Acceptance Criteria

1. WHEN designing the model THEN it SHALL use causal (masked) attention to prevent future information leakage
2. WHEN processing sequences THEN the model SHALL maintain temporal order and causality
3. WHEN encoding time information THEN the model SHALL use sinusoidal positional encoding appropriate for time series
4. WHEN predicting THEN the model SHALL generate outputs autoregressively step-by-step
5. WHEN training THEN the model SHALL use appropriate loss functions for time series regression

### Requirement 2: Essential Time Series Features and Normalization

**User Story:** As a data scientist, I want proper feature engineering and normalization following time series best practices, so that the model can learn meaningful patterns from the data.

#### Acceptance Criteria

1. WHEN preprocessing values THEN the system SHALL apply per-series standardization (z-score normalization)
2. WHEN creating time features THEN the system SHALL include cyclical encodings (sin/cos) for periodic patterns
3. WHEN engineering features THEN the system SHALL add lag features and rolling statistics
4. WHEN normalizing THEN the system SHALL preserve temporal relationships and avoid data leakage
5. WHEN denormalizing THEN the system SHALL correctly restore original scale using stored statistics

### Requirement 3: Time Series Training Best Practices

**User Story:** As a data scientist, I want training procedures that follow time series forecasting best practices, so that the model learns to generalize to future unseen data.

#### Acceptance Criteria

1. WHEN splitting data THEN the system SHALL use temporal splits (not random) to avoid data leakage
2. WHEN training THEN the system SHALL implement walk-forward validation for time series
3. WHEN optimizing THEN the system SHALL use appropriate learning rates and schedulers for sequential data
4. WHEN regularizing THEN the system SHALL apply dropout and weight decay suitable for time series
5. WHEN validating THEN the system SHALL evaluate on truly future data points

### Requirement 4: Multi-Step Forecasting Implementation

**User Story:** As a data scientist, I want proper multi-step ahead forecasting, so that the model can predict multiple future time steps accurately.

#### Acceptance Criteria

1. WHEN predicting multiple steps THEN the system SHALL use autoregressive generation (not teacher forcing)
2. WHEN generating sequences THEN the system SHALL feed previous predictions as input for next steps
3. WHEN training THEN the system SHALL use scheduled sampling to bridge training/inference gap
4. WHEN evaluating THEN the system SHALL measure performance at each prediction horizon
5. WHEN forecasting THEN the system SHALL handle uncertainty propagation across time steps

### Requirement 5: Time Series Evaluation Metrics

**User Story:** As a data scientist, I want evaluation metrics specific to time series forecasting, so that I can properly assess model performance and compare with baselines.

#### Acceptance Criteria

1. WHEN evaluating THEN the system SHALL compute MASE (Mean Absolute Scaled Error) against naive baselines
2. WHEN measuring accuracy THEN the system SHALL include directional accuracy and trend prediction
3. WHEN assessing performance THEN the system SHALL evaluate at different forecasting horizons
4. WHEN comparing models THEN the system SHALL use time series cross-validation
5. WHEN reporting results THEN the system SHALL include prediction intervals and uncertainty quantification