# Stage 1 fidelity (cub)

teacher: `cub/teacher/seed0/best.pt`; student for maskedkd / rollout: `/home/coder/stage0/outputs/cub/kd/seed0/last.pt` (eval mode).
Cell: **agree** (masked-teacher argmax = full-teacher argmax, %) / **KL**(p_full ‖ p_masked), τ=1. random: mean ± std over seeds 0, 1, 2.

Teacher GFLOPs per image (DeiT convention): keep 1 (k=196) = 17.56, keep 0.7 (k=137) = 12.19, keep 0.5 (k=98) = 8.70, keep 0.3 (k=59) = 5.28, keep 0.15 (k=29) = 2.68.  
Selection cost (GFLOPs/img beyond the student forward): maskedkd 0.022, rollout 0.435, random 0.000, teacher_cache 0.000, teacher_oracle 22.188.

## train_view (N=5394)

| criterion | keep 1 | keep 0.7 | keep 0.5 | keep 0.3 | keep 0.15 |
|---|---|---|---|---|---|
| maskedkd | 100.0 / 0.000 | 95.5 / 0.018 | 91.2 / 0.074 | 78.4 / 0.364 | 48.1 / 1.453 |
| rollout | 100.0 / 0.000 | 96.4 / 0.012 | 92.7 / 0.048 | 85.0 / 0.210 | 61.7 / 0.959 |
| random | 100.0 / 0.000 | 90.2 ± 0.6 / 0.094 ± 0.002 | 81.7 ± 0.3 / 0.276 ± 0.004 | 60.5 ± 0.5 / 0.945 ± 0.006 | 24.1 ± 0.6 / 2.555 ± 0.011 |
| teacher_cache:attn_last | 100.0 / 0.000 | 94.0 / 0.044 | 88.3 / 0.145 | 75.5 / 0.471 | 49.8 / 1.383 |
| teacher_cache:rollout | 100.0 / 0.000 | 93.0 / 0.057 | 85.9 / 0.181 | 73.2 / 0.552 | 45.8 / 1.561 |
| teacher_oracle:attn_last | 100.0 / 0.000 | 96.3 / 0.013 | 92.3 / 0.060 | 82.2 / 0.268 | 55.4 / 1.155 |
| teacher_oracle:rollout | 100.0 / 0.000 | 95.3 / 0.021 | 90.8 / 0.086 | 78.1 / 0.402 | 47.8 / 1.478 |

Full-teacher accuracy on this view: 84.72%.

Cache approximation (top-k overlap of teacher_cache with teacher_oracle, same kind): attn_last keep 0.7 78.0%; attn_last keep 0.5 67.3%; attn_last keep 0.3 57.3%; attn_last keep 0.15 43.2%; rollout keep 0.7 76.8%; rollout keep 0.5 64.5%; rollout keep 0.3 53.0%; rollout keep 0.15 42.6%.

## test_clean (N=5794)

| criterion | keep 1 | keep 0.7 | keep 0.5 | keep 0.3 | keep 0.15 |
|---|---|---|---|---|---|
| maskedkd | 100.0 / 0.000 | 96.9 / 0.013 | 93.3 / 0.047 | 85.9 / 0.216 | 56.7 / 1.252 |
| rollout | 100.0 / 0.000 | 97.4 / 0.007 | 95.0 / 0.028 | 88.9 / 0.136 | 70.8 / 0.743 |
| random | 100.0 / 0.000 | 91.6 ± 0.2 / 0.082 ± 0.003 | 83.9 ± 0.4 / 0.239 ± 0.004 | 65.3 ± 0.2 / 0.824 ± 0.008 | 28.7 ± 0.5 / 2.458 ± 0.014 |
| teacher_oracle:attn_last | 100.0 / 0.000 | 97.2 / 0.009 | 93.9 / 0.039 | 87.0 / 0.176 | 64.5 / 0.912 |
| teacher_oracle:rollout | 100.0 / 0.000 | 96.2 / 0.018 | 91.6 / 0.072 | 81.7 / 0.331 | 54.3 / 1.359 |

Full-teacher accuracy on this view: 84.88%.

