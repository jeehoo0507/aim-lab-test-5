# Stage 2 summary: MaskedKD / Random teacher-token reduction vs Full KD

Test = clean top-1 at the last epoch (mean ± std over seeds). "vs Full KD" = mean difference; per-seed differences pair runs with the same seed (same data order, augmentation and mixup). Teacher GFLOPs: one DeiT-B forward per image with k = round(keep·196) patch tokens (DeiT convention, FlopCounterMode/2).

## CUB

| 기준 | 비율 | teacher GFLOPs/img | test acc (mean ± std) | Full KD 대비 | seed별 차이 (vs KD) | corruption acc |
|---|---|---|---|---|---|---|
| Full KD | 1.0 | 17.56 | 81.73 ± 0.28 | 0 | | 60.46 ± 0.43 |
| maskedkd | 0.5 | 8.70 | 81.88 ± 0.24 | +0.14 | s0 +0.19, s1 +0.12, s2 +0.12 | 60.29 ± 0.61 |
| maskedkd | 0.3 | 5.28 | 81.77 ± 0.47 | +0.04 | s0 -0.12, s1 +0.29, s2 -0.05 | 58.96 ± 0.45 |
| maskedkd | 0.15 | 2.68 | 80.34 ± 0.36 | -1.39 | s0 -1.31, s1 -1.19, s2 -1.67 | 55.71 ± 0.32 |
| random | 0.5 | 8.70 | 81.04 ± 0.38 | -0.69 | s0 -0.55, s1 -0.47, s2 -1.05 | 58.62 ± 0.54 |
| random | 0.3 | 5.28 | 79.60 ± 0.46 | -2.13 | s0 -2.30, s1 -1.92, s2 -2.19 | 55.11 ± 0.43 |
| random | 0.15 | 2.68 | 77.62 ± 0.48 | -4.11 | s0 -4.33, s1 -3.95, s2 -4.06 | 51.50 ± 0.37 |
| CE | – | 0 | 80.12 ± 0.20 | -1.62 | s0 -1.52, s1 -1.67, s2 -1.66 | 56.18 ± 0.38 |

### maskedkd − random (same seed, same keep)

- keep 0.5: mean +0.83%p (s0 +0.74, s1 +0.59, s2 +1.17)
- keep 0.3: mean +2.17%p (s0 +2.17, s1 +2.21, s2 +2.14)
- keep 0.15: mean +2.72%p (s0 +3.02, s1 +2.76, s2 +2.38)

### 여유 구간 판정 (사전 고정: Full KD 대비 maskedkd 평균 −0.5%p 이상 하락하는 가장 큰 비율)

- **CUB: MaskedKD 한계 비율 = 0.15** (Full KD 대비 -1.39%p)

