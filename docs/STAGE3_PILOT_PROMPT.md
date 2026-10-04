# Stage 3 pilot: TAM (Teachability-Aware Masking) vs MaskedKD — cv 서버, 빠른 확인용

## 목적
본 실험 전에 TAM이 MaskedKD보다 나은 방향인지 **빠르게** 확인한다. 같은 서버(cv)에서 MaskedKD와 TAM을
같은 토큰 비율·같은 seed로 돌려 비교한다. 결론은 "가능성 있음 / 없음" 수준만 내고, 확정 비교는 Stage 3 본 실험에서 한다.

TAM이 졌을 때 원인이 **아이디어**인지 **teacher attribution 캐시의 부정확함**인지 구분할 수 있게 설계한다 (아래 0절).

## 0. Stage 1 결과 (cv2, CUB, 학습 끝난 KD student 기준) — 이 설계의 근거

test_clean agree (masked-teacher 1등 = full-teacher 1등, %):

| 기준 | keep 0.5 | keep 0.3 | keep 0.15 |
|---|---|---|---|
| maskedkd (student 마지막 블록 CLS attn) | 93.3 | 85.9 | 56.7 |
| rollout (student rollout) | 95.0 | 88.9 | **70.8** |
| random | 83.9 | 65.3 | 28.7 |
| teacher_oracle:attn_last (그 view에서 teacher 직접) | 93.9 | 87.0 | 64.5 |
| teacher_oracle:rollout | 91.6 | 81.7 | 54.3 |

train_view에서 teacher_cache:attn_last는 keep 0.3 75.5 / 0.15 49.8 (maskedkd 78.4 / 48.1).
캐시 top-k와 oracle top-k의 겹침: attn_last 0.3 → 57.3%, 0.15 → 43.2%.

여기서 나온 결정:
1. **비율:** 무너지는 구간은 0.3~0.15. 파일럿은 0.3, 0.15만. 0.1 이하는 하지 않는다.
2. **teacher 신호 종류:** attn_last가 rollout보다 낫다 (oracle, cache 모두). TAM 기본값은 `attn_last`.
3. **캐시가 부정확하다** (overlap 43~57%). TAM의 gap 토큰 상당수가 캐시 오차일 수 있다 → `tam_oracle` 상한선 run과 캐시 진단을 추가한다.
4. **student rollout이 가장 강한 경쟁자**다 → `rollout` 학습 baseline을 추가한다.
5. TAM의 student 신호 S는 MaskedKD와 같은 마지막 블록 CLS attn으로 둔다 (MaskedKD 대비 차이가 "teacher 신호를 섞은 효과"만 되게).

## 작업 규칙
- `stage12` 브랜치에서 새 브랜치 `stage3-pilot`을 만든다. Stage 0·1·2 동작은 바꾸지 않는다
  (기존 기준 `maskedkd`, `random`의 선택 결과와 학습 경로가 비트 단위로 같아야 한다 — 테스트로 확인).
- 기존 인프라(스케줄러 `--stage 2`, 재개, 평가, `summarize_stage2`)를 재사용한다. 새 기준은 `TRAIN_CRITERIA`에 추가만 한다.
- 짝맞춤 유지: 새 기준도 전역 RNG를 쓰지 않는다. `kd` / `maskedkd` / `rollout` / `tam*` 같은 seed에서 입력 배치·mixup 해시가 같아야 한다.

---

## 1. TAM 토큰 선택 (① 어떤 토큰)

입력 (micro-batch마다):
- `T` (B, 196): teacher attribution.
  - `tam`, `tam_var`: 캐시 → 현재 크롭·flip에 맞게 변환 (`attribution.crop_maps`). 추가 teacher 연산 없음.
  - `tam_oracle`: 현재 view(mixup 전)에서 teacher를 전체 토큰으로 한 번 더 돌려 구한 attn_last (`masking.py`의 `teacher_oracle`과 같은 계산). **진단용**이며 비용이 Full KD보다 크다.
  - kind 기본 `attn_last` (`--tam-kind rollout` 선택 가능)
- `S` (B, 196): student 마지막 블록 CLS→patch attention (MaskedKD와 같은 신호, 학습 forward에서 hook으로 얻음)

선택 (이미지별, 남길 개수 k):
1. **teacher 후보 풀:** `T` 상위 `m = min(196, 2k)`개. 나머지는 **pruned**.
2. **shared:** 풀 안에서 `S`가 높은 순으로 `k_shared = round((1 − g)·k)`개.
3. **gap:** 풀에서 shared를 뺀 나머지 중 `T`가 높은 순으로 `k_gap = k − k_shared`개.
4. 남길 토큰 = shared ∪ gap (정확히 k개, 중복 없음).

gap 비율 g (curriculum):
- `g(epoch) = g0 + (g1 − g0) · epoch / (epochs − 1)`, 기본 **g0 = 0.1, g1 = 0.5**
- 인자: `--tam-gap g0 g1` (같은 값 두 개면 고정 비율)

캐시 lookup에는 이미지 경로가 필요하므로 train loader가 (image, box)와 이미지 인덱스를 함께 넘기게 한다
(`build_train_transform(return_box=True)` + `numpy_collate_box`). **이미지 값은 기존과 비트 단위로 같아야 한다** (이미 테스트 있음, tam 모드에도 적용).
mixup/cutmix: 선택은 mixup 전 원본 view의 박스 기준으로 한다. 이 한계를 문서에 적는다.

## 2. 이미지별 토큰 수 (② 얼마나) — 가변 길이 처리

**결정: 3단계 버킷 방식.** 패딩·attention mask는 쓰지 않는다 (패딩하면 가장 긴 이미지 기준으로 연산해서 절감이 사라지고, timm 블록에 mask를 넣어야 함).

1. **집중도:** 이미지별로 크롭된 `T`를 합이 1이 되게 정규화하고, 상위 k개 토큰의 질량 `c = sum(top-k T)`를 집중도로 쓴다.
2. **버킷:** micro-batch 안에서 `c`로 정렬해 3등분한다.
   - 집중도 높은 1/3 → `k_lo = round(k·(1 − δ))`
   - 중간 1/3 → `k`
   - 집중도 낮은 1/3 → `k_hi = round(k·(1 + δ))`
   - 기본 **δ = 0.33**. 3등분이 정확히 나눠지지 않으면 남는 이미지는 중간 버킷에 넣는다 (평균 토큰 수를 로그에 기록하고 |평균 − k| ≤ 1을 assert).
3. **teacher forward:** 버킷마다 따로 `teacher_forward(teacher, x[b], keep_idx[b])`를 호출하고, logit을 원래 순서로 되돌려 붙인다 (ViT는 BatchNorm이 없어 이미지별 결과가 같음 — 이미 테스트 있음).
4. **FLOPs 보고:** 이미지당 teacher GFLOPs = 세 버킷 GFLOPs의 이미지 수 가중 평균 (`flops.teacher_gflops`로 k_lo, k, k_hi 각각 측정).

인자: `--tam-budget {fixed,bucket}`, `--tam-delta 0.33`.

## 3. 학습 모드

`train.py --mode maskedkd --mask-criterion {rollout,tam,tam_var,tam_oracle} --keep K` (출력 `cub/{criterion}_k{keep}/seed{s}/`):
- `rollout`: student rollout 상위 k개 (Stage 1의 `rollout` 기준을 학습에 연결). 매 step student 모든 블록의 attention이 필요하므로 hook을 전 블록에 건다. 선택 비용(GFLOPs/img)을 config와 요약에 기록한다.
- `tam`: ①만, 캐시 T, 모든 이미지 k개 고정
- `tam_var`: ① + ②, 캐시 T
- `tam_oracle`: ①만, oracle T (진단용 상한선)
- 나머지 설정은 Stage 2와 완전히 같다 (recipe, α=0.5, τ=1, lr, teacher, 100에포치, ckpt_e{10,30,60}).
- log.csv 추가: `mask_agree`(기존), `tam_gap_ratio`, `mean_k`, `bucket_k`, 그리고 tam 계열은 `cache_oracle_overlap`(매 에포치 첫 배치 1개에서만 oracle을 추가로 돌려 캐시 top-k와의 겹침을 기록, `tam`·`tam_var`만).
- config에 `tam_gap`, `tam_kind`, `tam_budget`, `tam_delta`, `tam_teacher_signal`(cache/oracle) 기록.

## 4. 캐시 진단 (학습 없음, 몇 분)

`stage0/diagnose_cache.py --dataset cub` → `results/stage3_cache_diag.{csv,md}`
- train split에서 이미지당 RRC view 4개 (전용 generator, seed 고정)를 만들고, 각 view에서 캐시 변환 맵과 oracle 맵(attn_last)을 구한다.
- **크롭 면적 비율**(박스 면적 / 원본 면적) 구간 [0.08, 0.2), [0.2, 0.4), [0.4, 0.7), [0.7, 1.0]별로, keep 0.3 / 0.15의 top-k overlap 평균과 Spearman 상관을 낸다. flip 여부별로도 나눈다.
- 비교용으로 "원본 전체 이미지의 oracle 맵 vs 크롭 view oracle 맵의 겹침"도 구간별로 낸다 (크롭 변환 자체의 한계 = 같은 이미지라도 크롭이 바뀌면 teacher가 보는 곳이 바뀌는 정도).
- md에 한 줄 결론: 겹침이 작은 크롭에서만 낮은지(→ 해상도 문제, 다중 스케일 캐시로 개선 가능), 모든 구간에서 낮은지(→ 문맥 의존, 캐시 방식의 한계).
- 개선 방법은 이번에 구현하지 않는다. 결과만 낸다.

## 5. 파일럿 실행 (cv 서버)

사전: cv 서버의 CUB teacher로 attribution 캐시 생성 (`make_attribution_cache --dataset cub`, 1분 이내) → 캐시 진단.

| 기준 | 비율 | seed | run 수 |
|---|---|---|---|
| maskedkd | 0.3, 0.15 | 0, 1 | 4 |
| rollout | 0.3, 0.15 | 0, 1 | 4 |
| tam | 0.3, 0.15 | 0, 1 | 4 |
| tam_var | 0.3, 0.15 | 0, 1 | 4 |
| tam_oracle | 0.15 | 0, 1 | 2 |

총 18 run + 평가. A5000 1장(5 슬롯) 기준 4시간 안팎. 스케줄러는 `tam_oracle`을 마지막에 넣는다 (가장 느림).
비교는 **cv 서버 안에서만** 한다 (cv의 `kd/seed{0,1}`). cv2의 Stage 2 결과와 섞지 않는다 (teacher 체크포인트가 다름).

스케줄러: 기준별 비율이 다르므로 두 번 호출한다 (끝난 run은 건너뜀).
```
./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd rollout tam tam_var --keeps 0.3 0.15 --seeds 0 1
./run_stage0.sh --stage 2 --datasets cub --criteria tam_oracle --keeps 0.15 --seeds 0 1
```

## 6. 파일럿 요약 (`summarize_stage2`에 행 추가)

같은 표 형식에 `rollout`, `tam`, `tam_var`, `tam_oracle` 행을 추가한다. teacher GFLOPs 열 옆에 **선택 비용 열**을 둔다 (rollout ≈ 0.435, tam_oracle ≈ 22.2 + 마스킹 forward — `tam_oracle`은 표에 "진단용, 비용 비교 대상 아님"으로 표시).
덧붙일 것:
- 비율별, 같은 seed 짝 차이: `tam − maskedkd`, `tam_var − maskedkd`, `tam_oracle − maskedkd`, `tam − rollout`, `tam_var − rollout`
- `tam_var`의 실제 teacher GFLOPs/img (버킷 가중 평균)
- 학습 중 `cache_oracle_overlap`의 에포치별 평균 (캐시가 학습 중에도 이 정도인지)

**파일럿 판단 (사전 고정).** "이긴다" = 두 seed 모두 높고 평균 +0.3%p 이상.
| tam 또는 tam_var vs maskedkd | tam_oracle vs maskedkd (0.15) | 판단 |
|---|---|---|
| 이김 | – | 가능성 있음 → 본 실험(3 seed)으로 확장. rollout도 이기는지 함께 보고 |
| 못 이김 | 이김 | 아이디어는 유효, **캐시가 병목** → 캐시 진단 결과로 캐시 개선 후 재실험 |
| 못 이김 | 못 이김 | 가능성 낮음 → 선택 규칙(풀 크기, g 스케줄) 재검토 |

추가로 tam 계열이 maskedkd는 이기고 rollout에 지면, 요약에 "rollout 대비 우위 없음"을 명시한다 (본 실험에서 S를 rollout으로 바꾼 TAM을 검토할 근거).

## 7. 검증 체크리스트
- [ ] 기존 `maskedkd`·`random` 선택 결과 비트 동일 (stage12와 비교)
- [ ] rollout 학습 선택이 Stage 1 `rollout` 기준(같은 student, 같은 입력)과 같은 인덱스
- [ ] tam: 선택 개수 정확히 k, 중복 없음, 모든 인덱스가 teacher 후보 풀 안. g=0이면 풀 안 S 상위 k개, g=1이면 풀 안 T 상위 k개와 같음
- [ ] tam_oracle: T가 Stage 1 `teacher_oracle:attn_last`와 같은 맵 (같은 view에서)
- [ ] tam_var: 버킷 3개, |평균 − k| ≤ 1, 버킷별 forward를 합친 logit이 이미지별 단독 forward와 같음
- [ ] 짝맞춤: kd / maskedkd / rollout / tam / tam_var / tam_oracle 같은 seed에서 입력 배치·mixup 해시 동일
- [ ] FLOPs: tam_var 평균 GFLOPs가 같은 k의 고정 GFLOPs와 ±3% 이내
- [ ] 캐시 진단: 가짜 데이터에서 실행되고, oracle vs oracle(같은 view) overlap이 100%
- [ ] Stage 0·1·2 unit test, smoke test 그대로 통과. 새 smoke (`smoke_test_stage3_pilot.sh`): 가짜 데이터·CPU·`/dev/shm` 64MB로 캐시 진단 + rollout, tam, tam_var, tam_oracle 1에포치 run → 평가 → 요약
- [ ] smoke 통과 후 커밋·push, README에 cv 서버 실행 명령 추가

## 서버 실행 (README에 추가)
```bash
cd ~ && git clone -b stage3-pilot https://github.com/jeehoo0507/aim-lab-test-5 aim-lab-stage3 && cd aim-lab-stage3
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
setsid nohup bash -c 'uv run python -m stage0.make_attribution_cache --dataset cub && uv run python -m stage0.diagnose_cache --dataset cub && ./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd rollout tam tam_var --keeps 0.3 0.15 --seeds 0 1 && ./run_stage0.sh --stage 2 --datasets cub --criteria tam_oracle --keeps 0.15 --seeds 0 1' > $HOME/stage3_pilot.log 2>&1 < /dev/null &
```
