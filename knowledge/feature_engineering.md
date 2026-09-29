# Feature Engineering

- Define every feature relative to the prediction timestamp and verify that its source would be available at inference time.
- Fit vocabularies, encoders, selectors, and dimensionality reduction on training data or training folds only.
- Prefer pipelines that apply identical transformations during training, validation, testing, and production inference.
- Evaluate new features through cross-validation with an appropriate baseline and metric; additional complexity is not automatically useful.
- Treat high-cardinality categorical variables carefully and avoid target encoding without out-of-fold construction.
- Inspect highly correlated and duplicate features for redundancy, unstable coefficients, and unnecessary compute; correlation alone is not a deletion rule.
- Use one-hot encoding for suitable nominal categories, ordinal encoding only for meaningful order, and handle unknown inference categories explicitly.
- Standardize scale-sensitive models such as linear models, SVMs, PCA, and distance-based methods; tree models usually do not require feature scaling.
- Choose normalization only when bounded magnitude has a meaningful role; outliers can strongly compress min-max scaled values.
- Feature selection must be evaluated inside cross-validation and should preserve a simple baseline to show whether it adds value.
- Document feature meaning, units, valid ranges, missing-value behavior, and dependencies on upstream columns.
