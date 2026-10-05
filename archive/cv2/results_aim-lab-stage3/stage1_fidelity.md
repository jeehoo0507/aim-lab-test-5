# Stage 1 fidelity (waterbirds)

teacher: `waterbirds/teacher/seed0/best.pt`; student for maskedkd / rollout: `/home/coder/stage0/outputs/waterbirds/kd/seed0/last.pt` (eval mode).
Cell: **agree** (masked-teacher argmax = full-teacher argmax, %) / **KL**(p_full ‖ p_masked), τ=1. random: mean ± std over seeds 0, 1, 2.

Teacher GFLOPs per image (DeiT convention): keep 1 (k=196) = 17.56, keep 0.7 (k=137) = 12.19, keep 0.5 (k=98) = 8.70, keep 0.3 (k=59) = 5.28, keep 0.15 (k=29) = 2.68.  
Selection cost (GFLOPs/img beyond the student forward): maskedkd 0.022, rollout 0.435, random 0.000, teacher_cache 0.000, teacher_oracle 22.188.

## train_view (N=4795)

| criterion | keep 1 | keep 0.7 | keep 0.5 | keep 0.3 | keep 0.15 |
|---|---|---|---|---|---|
| maskedkd | 100.0 / 0.000 | 99.6 / 0.003 | 99.1 / 0.013 | 96.3 / 0.076 | 87.5 / 0.274 |
| rollout | 100.0 / 0.000 | 99.7 / 0.002 | 99.4 / 0.007 | 98.1 / 0.036 | 93.5 / 0.151 |
| random | 100.0 / 0.000 | 99.3 ± 0.2 / 0.010 ± 0.003 | 98.2 ± 0.1 / 0.034 ± 0.005 | 95.3 ± 0.2 / 0.105 ± 0.005 | 89.2 ± 0.3 / 0.238 ± 0.004 |
| teacher_cache:attn_last | 100.0 / 0.000 | 99.6 / 0.006 | 99.0 / 0.022 | 96.6 / 0.065 | 90.8 / 0.201 |
| teacher_cache:rollout | 100.0 / 0.000 | 99.5 / 0.007 | 98.4 / 0.027 | 95.7 / 0.084 | 88.8 / 0.231 |
| teacher_oracle:attn_last | 100.0 / 0.000 | 99.6 / 0.003 | 99.0 / 0.012 | 97.5 / 0.054 | 90.8 / 0.207 |
| teacher_oracle:rollout | 100.0 / 0.000 | 99.6 / 0.005 | 99.3 / 0.017 | 96.6 / 0.070 | 90.3 / 0.206 |

Full-teacher accuracy on this view: 98.85%.

Cache approximation (top-k overlap of teacher_cache with teacher_oracle, same kind): attn_last keep 0.7 76.7%; attn_last keep 0.5 64.8%; attn_last keep 0.3 53.4%; attn_last keep 0.15 39.1%; rollout keep 0.7 77.3%; rollout keep 0.5 64.9%; rollout keep 0.3 52.3%; rollout keep 0.15 40.2%.

## test_clean (N=5794)

| criterion | keep 1 | keep 0.7 | keep 0.5 | keep 0.3 | keep 0.15 |
|---|---|---|---|---|---|
| maskedkd | 100.0 / 0.000 | 98.9 / 0.010 | 97.8 / 0.029 | 95.1 / 0.105 | 86.5 / 0.332 |
| rollout | 100.0 / 0.000 | 99.0 / 0.008 | 97.8 / 0.027 | 95.8 / 0.086 | 91.6 / 0.204 |
| random | 100.0 / 0.000 | 97.1 ± 0.1 / 0.037 ± 0.001 | 94.6 ± 0.1 / 0.103 ± 0.004 | 88.5 ± 0.4 / 0.299 ± 0.008 | 78.1 ± 0.7 / 0.542 ± 0.008 |
| teacher_oracle:attn_last | 100.0 / 0.000 | 98.7 / 0.009 | 97.8 / 0.030 | 95.4 / 0.095 | 88.8 / 0.275 |
| teacher_oracle:rollout | 100.0 / 0.000 | 98.2 / 0.016 | 96.6 / 0.056 | 92.6 / 0.163 | 83.8 / 0.342 |

Full-teacher accuracy on this view: 93.63%.

