# Machine-learning project review guidance

- Data leakage: split before fitting imputers, scalers, encoders, feature selectors or resamplers. Fit on training data only; transform validation/test data with the fitted objects. Use a Pipeline to enforce this boundary.
- Train/test split: keep test data untouched until final evaluation. Use stratification for suitable classification tasks; use chronological splits for time series and group-aware splits where entities repeat. Fix and report a random seed when applicable.
- Preprocessing: choose transformations based on training data, document them, and apply the identical fitted pipeline to inference data. Never call `fit_transform` on the held-out set.
- Missing values: measure missingness by column and segment before choosing imputation or deletion. Fit imputers on training data only. `dropna` may remove many or systematically different records.
- Feature engineering: derive features without future information or target leakage. Fit learned encoders/selectors on training folds only and validate their benefit with an appropriate metric.
- Suggestions must distinguish evidence from possible risks. Quote original code only when the exact source snippet is available; otherwise say it was not supplied. Corrected code is illustrative and should use the project's actual variable names only when known.
