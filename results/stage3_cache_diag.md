# Stage 3 pilot: attribution cache diagnosis (cub, attn_last)

5394 train images × 4 RRC views (seed 0); oracle = teacher attn_last on the view.
Cell: top-k overlap % / Spearman. cache_vs_view = what tam uses; whole_vs_view = fresh whole-image map through the same crop mapping (crop-mapping limit; the gap to cache_vs_view is the cache's storage error).

Sanity (same view's oracle twice, first batch): overlap 100.0%.

## cache_vs_view

| group | n | keep 0.3 | keep 0.15 |
|---|---|---|---|
| all | 21576 | 66.4 / 0.546 | 53.2 / 0.546 |
| area [0.08,0.2) | 3935 | 56.3 / 0.430 | 39.2 / 0.430 |
| area [0.2,0.4) | 6566 | 66.3 / 0.551 | 50.4 / 0.551 |
| area [0.4,0.7) | 8561 | 70.0 / 0.580 | 58.7 / 0.580 |
| area [0.7,1.0] | 2514 | 70.2 / 0.600 | 63.8 / 0.600 |
| flip | 10887 | 66.1 / 0.541 | 52.9 / 0.541 |
| no flip | 10689 | 66.7 / 0.551 | 53.5 / 0.551 |

## whole_vs_view

| group | n | keep 0.3 | keep 0.15 |
|---|---|---|---|
| all | 21576 | 66.4 / 0.546 | 53.2 / 0.546 |
| area [0.08,0.2) | 3935 | 56.3 / 0.430 | 39.2 / 0.430 |
| area [0.2,0.4) | 6566 | 66.3 / 0.551 | 50.4 / 0.551 |
| area [0.4,0.7) | 8561 | 70.0 / 0.580 | 58.7 / 0.580 |
| area [0.7,1.0] | 2514 | 70.2 / 0.600 | 63.8 / 0.600 |
| flip | 10887 | 66.1 / 0.541 | 52.9 / 0.541 |
| no flip | 10689 | 66.7 / 0.551 | 53.5 / 0.551 |

## 결론

- **혼합** (keep 0.3: 큰 크롭 70.2%, 작은 크롭 56.3%; 차이 15%p 미만).
- 판정 기준(사전 고정): cache_vs_view, keep 0.3. 가장 큰 구간 ≥ 70% 이고 (큰 − 작은) ≥ 15%p → 해상도; 가장 큰 구간 < 70% → 문맥 의존; 그 외 혼합.
- 캐시 저장 오차 (whole_vs_view − cache_vs_view, 전체, keep 0.3): +0.0%p.
