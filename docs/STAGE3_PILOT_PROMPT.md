# Stage 3 pilot: TAM (Teachability-Aware Masking) vs MaskedKD — cv 서버, 빠른 확인용

## 목적
본 실험 전에 TAM이 MaskedKD보다 나은 방향인지 **빠르게** 확인한다. 같은 서버(cv)에서 MaskedKD와 TAM을
같은 토큰 비율·같은 seed로 돌려 비교한다. 결론은 "가능성 있음 / 없음" 수준만 내고, 확정 비교는 Stage 3 본 실험에서 한다.

## 작업 규칙
- `stage12` 브랜치에서 새 브랜치 `stage3-pilot`을 만든다. Stage 0·1·2 동작은 바꾸지 않는다
  (기존 기준 `maskedkd`, `random`의 선택 결과와 학습 경로가 비트 단위로 같아야 한다 — 테스트로 확인).
- 기존 인프라(스케줄러 `--stage 2`, 재개, 평가, `summarize_stage2`)를 재사용한다. 새 기준은 `TRAIN_CRITERIA`에 추가만 한다.
- 짝맞춤 유지: 새 기준도 전역 RNG를 쓰지 않는다. `kd` / `maskedkd` / `tam*` 같은 seed에서 입력 배치·mixup 해시가 같아야 한다.

---

## 1. TAM 토큰 선택 (① 어떤 토큰)

입력 (micro-batch마다, 추가 teacher 연산 없음):
- `T` (B, 196): teacher attribution 캐시 → 현재 크롭·flip에 맞게 변환 (`attribution.crop_maps`). kind는 `attn_last` (인자로 `rollout` 선택 가능)
- `S` (B, 196): student 마지막 블록 CLS→patch attention (MaskedKD와 같은 신호, 학습 forward에서 hook으로 얻음)

선택 (이미지별, 남길 개수 k):
1. **teacher 후보 풀:** `T` 상위 `m = min(196, 2k)`개. 나머지는 **pruned** (teacher가 중요하게 보지 않음).
2. **shared:** 풀 안에서 `S`가 높은 순으로 `k_shared = round((1 − g)·k)`개.
3. **gap:** 풀에서 shared를 뺀 나머지 중 `T`가 높은 순으로 `k_gap = k − k_shared`개. (shared가 S 상위를 가져갔으므로 gap은 teacher는 중요하게 보는데 student가 덜 본 토큰이 된다.)
4. 남길 토큰 = shared ∪ gap (정확히 k개, 중복 없음).

gap 비율 g (curriculum, 학습 진행에 따라):
- `g(epoch) = g0 + (g1 − g0) · epoch / (epochs − 1)`, 기본 **g0 = 0.1, g1 = 0.5** (초반엔 따라 하기 쉬운 shared 위주, 후반엔 gap을 늘림)
- 인자: `--tam-gap g0 g1` (같은 값 두 개면 고정 비율)

캐시 lookup에는 이미지 경로가 필요하므로 train loader가 (image, box)와 이미지 인덱스를 함께 넘기게 한다
(`build_train_transform(return_box=True)` + `numpy_collate_box`). **이미지 값은 기존과 비트 단위로 같아야 한다** (이미 테스트 있음, tam 모드에도 적용).
mixup/cutmix: 선택은 mixup 전 원본 view의 박스 기준으로 한다 (MaskedKD도 mixup 후 입력의 student attention을 쓰지만, 캐시는 원본 기준만 있으므로). 이 한계를 문서에 적는다.

## 2. 이미지별 토큰 수 (② 얼마나) — 가변 길이 처리

**결정: 3단계 버킷 방식.** 패딩·attention mask는 쓰지 않는다 (패딩하면 가장 긴 이미지 기준으로 연산해서 절감이 사라지고, timm 블록에 mask를 넣어야 함).

1. **집중도:** 이미지별로 크롭된 `T`를 합이 1이 되게 정규화하고, 상위 k개 토큰의 질량 `c = sum(top-k T)`를 집중도로 쓴다. (클수록 teacher attribution이 소수 토큰에 몰림 → 토큰이 덜 필요)
2. **버킷:** micro-batch 안에서 `c`로 정렬해 3등분한다.
   - 집중도 높은 1/3 → `k_lo = round(k·(1 − δ))`
   - 중간 1/3 → `k`
   - 집중도 낮은 1/3 → `k_hi = round(k·(1 + δ))`
   - 기본 **δ = 0.33**. 3등분이 정확히 나눠지지 않으면 남는 이미지는 중간 버킷에 넣어서, 이미지당 평균 토큰 수가 k에서 벗어나지 않게 한다 (평균을 로그에 기록하고 |평균 − k| ≤ 1을 assert).
3. **teacher forward:** 버킷마다 따로 `teacher_forward(teacher, x[b], keep_idx[b])`를 호출하고, logit을 원래 순서로 되돌려 붙인다. ViT에는 BatchNorm이 없어서 따로 돌려도 이미지별 결과는 같다 (이미 테스트 있음: per-sample 독립성).
4. **FLOPs 보고:** 이미지당 teacher GFLOPs = 세 버킷 GFLOPs의 (이미지 수 가중) 평균. attention이 토큰 수의 제곱이라 k 고정과 정확히 같지는 않으니, 실제 평균값을 계산해서 기록한다 (`flops.teacher_gflops`로 k_lo, k, k_hi 각각 측정).

인자: `--tam-budget {fixed,bucket}`, `--tam-delta 0.33`.

## 3. 학습 모드

`train.py --mode maskedkd --mask-criterion {tam,tam_var} --keep K` (출력 `cub/{criterion}_k{keep}/seed{s}/`):
- `tam`: ①만 (모든 이미지 k개 고정)
- `tam_var`: ① + ② (버킷 3단계, 평균 k개)
- 나머지 설정은 Stage 2와 완전히 같다 (recipe, α=0.5, τ=1, lr, teacher, 100에포치, ckpt_e{10,30,60}).
- log.csv에 추가: `mask_agree`(기존), `tam_gap_ratio`(그 에포치의 g), `mean_k`(tam_var의 실제 평균 토큰 수), `bucket_k`(k_lo/k/k_hi).
- config에 `tam_gap`, `tam_kind`, `tam_budget`, `tam_delta` 기록.

## 4. 파일럿 실행 (cv 서버)

사전: cv 서버의 CUB teacher로 attribution 캐시 생성 (`make_attribution_cache --dataset cub`, 1분 이내).

| 기준 | 비율 | seed | run 수 |
|---|---|---|---|
| maskedkd | 0.3, 0.15 | 0, 1 | 4 |
| tam | 0.3, 0.15 | 0, 1 | 4 |
| tam_var | 0.3, 0.15 | 0, 1 | 4 |

총 12 run + 평가. A5000 1장(5 슬롯) 기준 2.5~3시간 예상.
비교는 **cv 서버 안에서만** 한다 (cv의 `kd/seed{0,1}`, `maskedkd_k*/seed{0,1}`). cv2의 Stage 2 결과와 섞지 않는다 (teacher 체크포인트가 다름).

스케줄러: `./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd tam tam_var --keeps 0.3 0.15 --seeds 0 1`

## 5. 파일럿 요약 (`summarize_stage2`에 tam 행 추가)

같은 표 형식에 `tam`, `tam_var` 행을 추가하고, 아래를 덧붙인다:
- 비율별 `tam − maskedkd`, `tam_var − maskedkd` (같은 seed 짝 차이, seed별)
- `tam_var`의 실제 teacher GFLOPs/img (버킷 가중 평균)와 `maskedkd`와의 차이
- **파일럿 판단 (사전 고정):** 어떤 비율에서든 `tam` 또는 `tam_var`가 maskedkd보다 **두 seed 모두 높고 평균 +0.3%p 이상**이면 "가능성 있음 → 본 실험(3 seed)으로 확장", 아니면 "가능성 낮음 → 설계 재검토".

## 6. 검증 체크리스트
- [ ] 기존 `maskedkd`·`random` 선택 결과 비트 동일 (stage12와 비교)
- [ ] tam: 선택 개수 정확히 k, 중복 없음, 모든 인덱스가 teacher 후보 풀 안. g=0이면 풀 안 S 상위 k개, g=1이면 풀 안 T 상위 k개와 같음
- [ ] tam_var: 버킷 3개, 이미지당 평균 토큰 수 |평균 − k| ≤ 1, 버킷별 forward 결과를 합친 logit이 이미지별 단독 forward와 같음
- [ ] 짝맞춤: kd / maskedkd / tam / tam_var 같은 seed에서 입력 배치·mixup 해시 동일
- [ ] FLOPs: tam_var 평균 GFLOPs가 같은 k의 고정 GFLOPs와 ±3% 이내
- [ ] Stage 0·1·2 unit test, smoke test 그대로 통과. 새 smoke (`smoke_test_stage3_pilot.sh`): 가짜 데이터·CPU·`/dev/shm` 64MB로 tam, tam_var 1에포치 run → 평가 → 요약
- [ ] smoke 통과 후 커밋·push, README에 cv 서버 실행 명령 추가

## 서버 실행 (README에 추가)
```bash
cd ~ && git clone -b stage3-pilot https://github.com/jeehoo0507/aim-lab-test-5 aim-lab-stage3 && cd aim-lab-stage3
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
setsid nohup bash -c 'uv run python -m stage0.make_attribution_cache --dataset cub && ./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd tam tam_var --keeps 0.3 0.15 --seeds 0 1' > $HOME/stage3_pilot.log 2>&1 < /dev/null &
```
