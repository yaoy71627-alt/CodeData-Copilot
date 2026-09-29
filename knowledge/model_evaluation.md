# Model Evaluation

- For classification, choose among precision, recall, F1, ROC-AUC, PR-AUC, log loss, and calibration according to class balance and error cost; always include the confusion matrix or equivalent counts.
- For regression, pair MAE or median absolute error with RMSE when large errors matter, and interpret R-squared against a meaningful baseline.
- Compare against a simple baseline and report validation design, sample size, uncertainty, and meaningful segment performance.
- Use stratified, grouped, or time-aware splitting when the data-generating process requires it; random splitting is not universally valid.
- Cross-validation estimates variation across splits, but preprocessing and feature selection must be refitted inside every fold.
- Keep model selection and hyperparameter tuning separate from final test evaluation.
- Use training data to fit, validation data or cross-validation to choose models and thresholds, and the held-out test set once for final unbiased evaluation.
- Check overfitting by comparing training and validation behavior, preferably across multiple folds or time windows.
- Avoid a single-metric conclusion: inspect error slices, class-wise results, calibration, uncertainty, and operational constraints. Important subgroups can fail behind a strong aggregate score.
- Do not present correlation or offline metric improvement as causal or production impact without appropriate evidence.
