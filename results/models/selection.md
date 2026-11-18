# Model selection

Rule (fixed before results): highest student-test Top-1 (classifier only; HMM results not available yet); tiebreak lower student-test log loss; CV std and inference time also considered. Training accuracy is not used.

**Selected: `lgbm_tuned`** (Boosting): st_top1 = 0.9034, CV Top-1 = 0.8974 ± 0.0316, ST log loss = 0.08161667298781496.

| name | family | cv_macro_f1_mean | cv_macro_f1_std | cv_top1_mean | train_top1 | st_top1 | st_top1_hmm | st_macro_f1 | st_log_loss | infer_s_per_trip |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_tuned | Boosting | 0.9451 | 0.0126 | 0.8974 | 0.9379 | 0.9034 |  | 0.9421 | 0.0816 | 0.0271 |
| lgbm | Boosting | 0.9443 | 0.0128 | 0.8965 | 0.9389 | 0.9012 |  | 0.9405 | 0.0859 | 0.0552 |
| rf | Bagging ensemble | 0.9321 | 0.0162 | 0.8909 | 0.9079 | 0.8921 |  | 0.9254 | 0.1092 | 0.0331 |
| logreg_C1.0 | Linear | 0.8966 | 0.0097 | 0.8873 | 0.8876 | 0.8804 |  | 0.8837 | 0.1772 | 0.0023 |
| logreg_C0.1 | Linear | 0.8966 | 0.0098 | 0.8873 | 0.8877 | 0.8801 |  | 0.8837 | 0.1772 | 0.0023 |
| knn | Instance-based | 0.9234 | 0.0226 | 0.8744 | 0.982 | 0.8766 |  | 0.9205 | 0.1591 | 14.9471 |
| gnb_vs1e-06 | Probabilistic | 0.8583 | 0.0097 | 0.8484 | 0.8493 | 0.839 |  | 0.842 | 0.9888 | 0.0041 |
| gnb_vs1e-09 | Probabilistic | 0.8542 | 0.0107 | 0.8387 | 0.84 | 0.828 |  | 0.8393 | 1.1459 | 0.0042 |
| b0_nearest | Baseline (no ML) | 0.7306 | 0.0216 | 0.5485 | 0.5464 | 0.5147 |  | 0.7089 |  | 0.0002 |