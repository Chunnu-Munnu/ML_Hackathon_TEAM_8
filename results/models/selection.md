# Model selection

Rule (fixed before results): highest student-test Top-1 after HMM; tiebreak lower student-test log loss; CV std and inference time also considered. Training accuracy is not used.

**Selected: `rf`** (Bagging ensemble): st_top1_hmm = 0.9285, CV Top-1 = 0.8963 ± 0.0328, ST log loss = 0.10411293324267568.

| name | family | cv_macro_f1_mean | cv_macro_f1_std | cv_top1_mean | train_top1 | st_top1 | st_top1_hmm | st_macro_f1 | st_log_loss | infer_s_per_trip |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | Boosting | 0.9477 | 0.011 | 0.9015 | 0.9488 | 0.9065 | 0.9235 | 0.9441 | 0.0786 | 0.0413 |
| lgbm_tuned | Boosting | 0.9491 | 0.0112 | 0.9044 | 0.9533 | 0.9062 | 0.9206 | 0.9444 | 0.0779 | 0.1362 |
| mlp_128_64_32 | Neural network | 0.9401 | 0.0215 | 0.8941 | 0.9297 | 0.905 | 0.9242 | 0.9395 | 0.0937 | 0.008 |
| mlp_64_32 | Neural network | 0.9409 | 0.0213 | 0.8942 | 0.9199 | 0.902 | 0.9252 | 0.9386 | 0.0896 | 0.0066 |
| rf | Bagging ensemble | 0.9383 | 0.0164 | 0.8963 | 0.9148 | 0.8981 | 0.9285 | 0.9279 | 0.1041 | 0.0259 |
| dtree | Tree | 0.9274 | 0.0148 | 0.8957 | 0.9164 | 0.8891 | 0.9257 | 0.9152 | 0.1274 | 0.0024 |
| logreg_C1.0 | Linear | 0.9016 | 0.0097 | 0.8953 | 0.8953 | 0.8867 | 0.912 | 0.8901 | 0.1591 | 0.0032 |
| logreg_C0.1 | Linear | 0.9016 | 0.0097 | 0.8952 | 0.8952 | 0.8867 | 0.9116 | 0.89 | 0.1593 | 0.0032 |
| svm_linear_C1.0 | Kernel / margin | 0.9321 | 0.0152 | 0.8931 | 0.894 | 0.8862 | 0.9139 | 0.923 | 0.1151 | 0.012 |
| svm_linear_C0.1 | Kernel / margin | 0.9321 | 0.0152 | 0.8931 | 0.894 | 0.8862 | 0.9139 | 0.9229 | 0.1151 | 0.0132 |
| knn | Instance-based | 0.9288 | 0.0212 | 0.8825 | 0.982 | 0.8833 | 0.9096 | 0.9237 | 0.148 | 3.2079 |
| gnb_vs1e-06 | Probabilistic | 0.8763 | 0.0068 | 0.8704 | 0.8703 | 0.8598 | 0.889 | 0.8597 | 0.9405 | 0.0064 |
| gnb_vs1e-09 | Probabilistic | 0.875 | 0.0074 | 0.8611 | 0.8617 | 0.8492 | 0.8773 | 0.8599 | 1.0291 | 0.0063 |
| b0_nearest | Baseline (no ML) | 0.7306 | 0.0216 | 0.5485 | 0.5464 | 0.5147 | 0.5979 | 0.7089 |  | 0.0002 |