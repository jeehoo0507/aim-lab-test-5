# Stage 0 summary: CE vs Full KD

Values: 3-seed mean ± std (teacher: single run). Students at the last epoch; teacher at best val.
Corruption = mean accuracy over the cached conditions (15 corruptions × 5 severities) on a fixed test subset: 75 conditions × 1000 images.

| 데이터셋 | 방법 | Clean | WGA | Corruption |
|---|---|---|---|---|
| CUB | Teacher | 84.88 | – | 66.70 |
| CUB | CE | 80.12 ± 0.20 | – | 56.18 ± 0.38 |
| CUB | KD | 81.73 ± 0.28 | – | 60.46 ± 0.43 |
| Waterbirds | Teacher | 93.63 | 75.86 | 83.42 |
| Waterbirds | CE | 90.06 ± 0.55 | 71.91 ± 1.75 | 76.87 ± 0.80 |
| Waterbirds | KD | 90.97 ± 0.46 | 73.99 ± 1.87 | 78.72 ± 0.60 |

## Notes

- CUB teacher train_view: acc 84.72%, p_T(y) τ1 0.712 / τ4 0.044, wrong-class entropy/log(C−1) τ1 0.555 / τ4 0.986
- CUB teacher test_clean: acc 84.88%, p_T(y) τ1 0.736 / τ4 0.047, wrong-class entropy/log(C−1) τ1 0.490 / τ4 0.985
- Waterbirds teacher train_view: acc 98.85%, p_T(y) τ1 0.981 / τ4 0.873, wrong-class entropy/log(C−1) τ1 n/a (C=2)
- Waterbirds teacher test_clean: acc 93.63%, p_T(y) τ1 0.928 / τ4 0.830, wrong-class entropy/log(C−1) τ1 n/a (C=2)

## 판정

규칙: CUB=clean top-1, Waterbirds=WGA. 통과 = (KD−CE) 3-seed 평균 ≥ 1.0%p 그리고 같은 seed 차이 3개 모두 > 0. clean 실패 시 corruption 정확도로 같은 규칙 → OOD 통과.

- **CUB: 통과** — clean_acc: KD−CE = +1.62%p (per seed +1.52, +1.67, +1.66)
- **Waterbirds: 통과** — wga: KD−CE = +2.08%p (per seed +1.71, +2.49, +2.02)
