# Implementation Plan

- [ ] 1. Implement per-series normalization (CRITICAL - fixes prediction scale)
  - Create TimeSeriesNormalizer class with per-series z-score normalization
  - Add fit, transform, and inverse_transform methods that handle series_id
  - Implement proper statistics storage and retrieval per series
  - Write unit tests for normalization consistency and series isolation
  - _Requirements: 2.1, 2.4_

- [ ] 2. Create proper time series feature engineering
  - Implement TimeSeriesFeatureExtractor with cyclical sin/cos encoding for time features
  - Add lag features (previous values) and rolling statistics (mean, std)
  - Remove linear normalization of cyclical features, use proper cyclical encoding
  - Write unit tests for cyclical feature encoding and lag feature creation
  - _Requirements: 2.2, 2.3_

- [ ] 3. Implement causal transformer architecture for time series
  - Create TimeSeriesTransformer with causal (masked) self-attention to prevent future leakage
  - Add sinusoidal positional encoding appropriate for time series
  - Implement proper layer normalization and residual connections
  - Add dropout for regularization and appropriate model dimensions (64 dim, 4 heads, 3 layers)
  - Write unit tests for causal masking and forward pass
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 4. Implement temporal data splitting and training pipeline
  - Create TimeSeriesTrainer with temporal_train_val_test_split (no random shuffling)
  - Implement proper train/validation/test splits using temporal order (60/20/20)
  - Add walk-forward validation for time series cross-validation
  - Write unit tests for temporal splitting and data leakage prevention
  - _Requirements: 3.1, 3.2_

- [ ] 5. Implement autoregressive training and prediction
  - Add train_autoregressive method with step-by-step prediction training
  - Implement scheduled sampling to bridge training/inference gap
  - Create predict_autoregressive method for proper inference without teacher forcing
  - Add proper gradient clipping and learning rate scheduling for time series
  - Write unit tests for autoregressive generation and scheduled sampling
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 6. Create time series evaluation metrics
  - Implement MASE (Mean Absolute Scaled Error) against naive baselines
  - Add directional accuracy and trend prediction metrics
  - Create per-horizon evaluation (performance at each prediction step)
  - Implement time series cross-validation for model comparison
  - Write unit tests for all time series specific metrics
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 7. Enhance visualization for time series forecasting
  - Improve plot_series_df with proper time series formatting and seasonal decomposition
  - Enhance plot_predictions_df with per-horizon accuracy and confidence intervals
  - Add training curve plotting with proper time series validation curves
  - Create residual analysis plots and prediction distribution comparisons
  - Write unit tests for all enhanced plotting functions
  - _Requirements: 5.5_

- [ ] 8. Integrate time series components into data pipeline
  - Modify SignalDataset to use TimeSeriesNormalizer and TimeSeriesFeatureExtractor
  - Update SignalDataLoader to handle temporal splitting and per-series processing
  - Ensure backward compatibility while adding time series best practices
  - Add proper state saving/loading for normalizer statistics
  - Write integration tests for complete time series pipeline
  - _Requirements: 2.1, 2.4, 3.1_

- [ ] 9. Update training script with time series best practices
  - Modify train.py to use TimeSeriesTrainer and TimeSeriesTransformer
  - Add command-line arguments for time series specific parameters
  - Implement proper model checkpointing with denormalization capability
  - Add comprehensive logging of time series metrics and validation
  - Write integration tests comparing old vs new model performance
  - _Requirements: 1.1, 3.1, 4.1_

- [ ] 10. Create comprehensive time series examples and documentation
  - Update example.py to demonstrate proper time series forecasting workflow
  - Create documentation explaining time series best practices implemented
  - Add example showing before/after prediction quality improvements
  - Update README with time series forecasting capabilities and expected performance
  - Create troubleshooting guide for time series specific issues
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_