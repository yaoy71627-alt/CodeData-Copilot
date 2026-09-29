# sklearn and pandas Best Practices

- Use `Pipeline` and `ColumnTransformer` to bind preprocessing to the estimator and keep training, cross-validation, testing, and inference transformations consistent.
- Call `fit` or `fit_transform` only on training data; call `transform` on validation, test, and production inputs.
- Pass `random_state` to train/test split, stochastic estimators, samplers, and search procedures when reproducibility is required.
- Use stratification only when it is statistically appropriate and every class has enough samples; use grouped or temporal splitters for dependent observations.
- Configure encoders to handle unknown categories at inference time and keep the learned category vocabulary with the fitted pipeline.
- In pandas, prefer `.loc` assignments or an explicit `.copy()` to avoid ambiguous chained assignment and accidental view mutation.
- Validate join keys and use merge cardinality checks such as `validate='one_to_one'` or `validate='many_to_one'` when the expected grain is known.
- Avoid silent dtype inference for critical identifiers, dates, and categories; declare or validate dtypes at ingestion.
- Make `read_csv` and output operations explicit about encoding, separators, index columns, missing sentinels, and destination paths.
- Avoid repeated row-wise `apply` for large data when vectorized operations are clearer and faster, but verify semantic equivalence before optimization.
- Keep target columns out of generic feature transformations and ensure pandas index alignment does not silently pair the wrong labels and features.
- When using cross-validation utilities, supply the complete pipeline rather than precomputed features created from the full dataset.
