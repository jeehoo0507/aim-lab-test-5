# Stage 1 + 2: teacher 토큰 마스킹, fidelity 측정, MaskedKD baseline

## 배경
Stage 0에서 CUB가 게이트를 통과했다 (test, 3 seed). 이제 이 환경에서 teacher 입력 토큰을 줄이는 실험을 한다.

| | CE | Full KD | KD − CE |
|---|---|---|---|
| CUB test (seed 0/1/2) | 79.89 / 80.27 / 80.19 | 81.41 / 81.95 / 81.84 | 평균 +1.62%p |

- **Stage 1 (학습 없음):** 마스킹 기준 × 남길 토큰 비율별로, 토큰을 줄인 teacher의 출력이 원래 teacher와 얼마나 같은지(fidelity) 잰다.
- **Stage 2 (학습):** MaskedKD와 Random 마스킹으로 student를 학습해서, Full KD 대비 정확도가 떨어지기 시작하는 토큰 비율(우리 방법이 이길 여유 공간)을 찾는다.
- Stage 3(제안 방법 TAM)은 이번 범위가 아니다. 다만 teacher attribution 캐시와 크롭 좌표 변환은 Stage 1에서 만들어 Stage 3에서 재사용한다.

---

## 0. 작업 규칙 (중요)

- **Stage 0이 서버의 `~/aim-lab-test-5`에서 아직 돌고 있다.** 실행 중인 스케줄러가 새 학습·평가 프로세스를 띄울 때마다 그 폴더의 코드를 import하므로, 그 폴더나 `claude/upbeat-curie-lhrxvg` 브랜치를 건드리면 안 된다.
- 현재 브랜치에서 **새 브랜치 `stage12`**를 만들어 작업한다. 서버에서는 별도 폴더(`~/aim-lab-stage12`)에 clone해서 실행한다 (README에 명령 추가).
- Stage 0 코드의 동작은 바꾸지 않는다. 기존 함수에 인자를 추가할 때는 기본값이 기존 동작과 완전히 같아야 하고, 이를 테스트로 확인한다.
- Stage 0 결과(`$OUTPUT_ROOT/cub/{teacher,ce,kd}/...`)는 **읽기만** 한다. 새 결과는 새 폴더에만 쓴다.
- 기존 인프라를 그대로 쓴다: 스케줄러, 재개(last.pt), DONE/FAILED, numpy collate(`/dev/shm` 회피), cgroup-aware CPU·RAM, preflight.
- 데이터셋은 CUB만 대상으로 한다. 코드는 dataset 인자를 받되, Waterbirds 실행은 하지 않는다.

---

## 1. teacher 토큰 마스킹

`stage0/models.py`의 `teacher_forward(teacher, images, keep_idx=None)`:
- `keep_idx`: `(B, k)` long tensor. patch 토큰 인덱스(0..195). `None`이면 기존과 동일하게 전체 토큰을 쓴다.
- `_pos_embed` 다음에 cls 토큰 + 고른 patch 토큰만 남긴다 (MaskedKD `models_teacher.py:261-266`와 같은 위치):
  `x = cat([x[:, :1], gather(x[:, 1:], keep_idx)], dim=1)`
- `teacher.num_prefix_tokens == 1`을 assert한다 (DeiT는 cls 1개, register 토큰 없음).
- 테스트:
  - `keep_idx=None`과 `keep_idx=arange(196)`의 출력이 `teacher(x)`와 같다 (`torch.equal`)
  - 토큰 순서를 섞어도 출력이 같다 (ViT는 위치 임베딩을 미리 더해서 순서 무관). 허용 오차 1e-5
  - k가 줄면 FLOPs가 줄어든다 (아래 4절 FLOPs 측정으로 확인)

---

## 2. 마스킹 기준 (`stage0/masking.py`)

`k = round(keep × 196)`개를 남긴다. 모든 기준은 `(B, k)` 인덱스를 반환한다.

| 이름 | 신호 | 비용 | 계산 |
|---|---|---|---|
| `maskedkd` | student 마지막 블록 attention의 CLS→patch 행, head 평균, top-k | 공짜 (student forward 재사용) | MaskedKD `losses.py:32`와 동일 |
| `random` | 균일 무작위 k개 | 0 | 전용 `torch.Generator`로 뽑는다 (아래 주의) |
| `rollout` | student 전 레이어 attention rollout의 CLS 행, top-k | 공짜에 가까움 (student attention 재사용) | `R = Π (0.5·A_l + 0.5·I)`, A_l은 head 평균 |
| `teacher_cache` | 학습 전 저장한 teacher attribution을 현재 크롭에 맞게 변환, top-k | 상각 (이미지당 1회) | 3절 |
| `teacher_oracle` | 현재 view에서 teacher를 직접 돌려 얻은 attribution, top-k | 비쌈 (Stage 1 상한 참고용, 학습에는 쓰지 않음) | 3절과 같은 attribution을 그 view에서 계산 |

**student attention 얻기:** timm DeiT는 fused attention(SDPA)이라 attention 가중치가 밖으로 나오지 않는다.
- 블록의 `attn` 모듈에 forward hook을 걸고, 입력(`norm1` 이후)에서 `qkv`로 q, k를 다시 계산해 `softmax(q·kᵀ·scale)`를 구한다. student forward의 출력은 바꾸지 않는다.
- `maskedkd`는 마지막 블록만, `rollout`은 전 블록에 hook을 건다. 필요할 때만 hook을 등록해서, 다른 모드의 속도와 수치에 영향이 없게 한다.
- 학습 중에는 student의 학습용 forward(train 모드, drop_path 켜짐)에서 나온 attention을 쓴다 (MaskedKD와 동일).
- 테스트: hook으로 구한 attention과, `fused_attn=False`로 바꿔 직접 계산한 attention이 같아야 한다.

**주의 (짝맞춤 유지):** `random` 기준이 전역 RNG를 쓰면 mixup·drop_path 난수 순서가 바뀌어, 같은 seed의 KD run과 데이터·증강이 달라진다. `torch.Generator().manual_seed(C.epoch_seed(seed, epoch, 4) + step)` 같은 전용 generator만 써라. 테스트로 확인한다: 같은 seed에서 `kd`와 `maskedkd`/`random` run의 student 입력 배치와 mixup 타깃 해시가 같아야 한다 (Stage 0 리뷰 때 쓴 방식).

---

## 3. teacher attribution 캐시 + 크롭 좌표 변환 (`stage0/attribution.py`)

**캐시 생성 (`stage0/make_attribution_cache.py`, 학습 전 1회)**
- 대상: CUB train split 전체(5,394장). 입력: 원본 이미지 전체를 224×224로 리사이즈(크롭 없음).
- teacher(`$OUTPUT_ROOT/cub/teacher/seed0/best.pt`)로 두 가지 attribution을 14×14 맵으로 저장한다:
  - `attn_last`: 마지막 블록 CLS→patch attention, head 평균
  - `rollout`: teacher attention rollout의 CLS 행
- 저장: `$OUTPUT_ROOT/cub/attribution_cache/{attn_last,rollout}.npy` (float16, `[N, 14, 14]`, split 순서 그대로) + `manifest.json` (teacher run_uid, split 해시, 생성 시각). teacher가 바뀌었으면(run_uid 불일치) 로드를 거부한다.

**크롭 좌표 변환**
- train transform이 각 샘플의 RRC 박스(원본 좌표 `i, j, h, w`)와 flip 여부를 함께 반환하도록, RRC를 감싼 transform을 만든다. **반환하는 이미지 값은 기존과 비트 단위로 같아야 한다** (같은 seed에서 기존 transform과 비교하는 테스트).
- 맵 변환: 14×14 맵을 원본 크기로 보고 그 박스 영역을 잘라 14×14로 리샘플링(bilinear, `roi_align` 또는 `grid_sample`)하고, flip이면 좌우를 뒤집는다.
- 한계를 문서에 적는다: RandAugment의 기하 변환(회전·전단·평행이동)과 mixup/cutmix는 반영하지 않는다. Stage 1에서 `teacher_cache`와 `teacher_oracle`을 비교해서 이 근사의 오차를 잰다.
- Stage 2는 이 캐시를 쓰지 않는다.

---

## 4. Stage 1: fidelity 측정 (`stage0/fidelity.py`, 학습 없음)

**설정**
- teacher: `cub/teacher/seed0/best.pt`
- student 기반 기준(`maskedkd`, `rollout`)에 쓸 student: `cub/kd/seed0/last.pt` (학습 끝). Stage 2가 끝나면 중간 체크포인트(5절)로도 다시 잴 수 있게 `--student-ckpt` 인자를 둔다.
- 기준: `maskedkd`, `random`, `rollout`, `teacher_cache`(attn_last, rollout 두 종류), `teacher_oracle`
- 남길 토큰 비율: 1.0(기준점), 0.7, 0.5, 0.3, 0.15
- view 두 가지:
  - `train_view`: train split, train transform(mixup 전), 고정 seed, 1 pass
  - `test_clean`: test split, Resize(256) → CenterCrop(224)
  - `random`은 seed 3개로 반복해서 평균과 표준편차를 낸다.

**지표 (각 기준 × 비율 × view)**
- `kl`: KL(p_full ‖ p_masked), τ=1, 샘플 평균
- `agree`: argmax(p_masked) == argmax(p_full) 비율
- `acc`: argmax(p_masked) == 정답 비율
- `teacher_gflops`: 그 토큰 수에서 teacher forward 1회의 GFLOPs (`torch.utils.flop_counter.FlopCounterMode`로 실측). 기준 계산 비용(student attention 재계산, rollout, 맵 변환)도 따로 잰다.

**출력**
- `results/stage1_fidelity.csv` (긴 형식: criterion, keep, view, seed, kl, agree, acc, teacher_gflops, criterion_gflops)
- `results/stage1_fidelity.md`: view별로 표 하나씩 (행: 기준, 열: 비율, 칸: `agree / kl`)
- `results/stage1_fidelity.png`: x=남길 토큰 비율, y=agree, 기준별 선. view별 subplot

---

## 5. Stage 2: MaskedKD / Random baseline 학습

**train.py에 모드 추가**
- `--mode maskedkd --mask-criterion {maskedkd,random} --keep {0.5,0.3,0.15}`
- 나머지 설정은 Stage 0 KD와 완전히 같다: 같은 recipe(student 증강·mixup·LS 0.1), α=0.5, τ=1, CUB에서 고른 student lr(`lr_selection.json`), 100에포치, 같은 teacher.
- 출력 경로: `$OUTPUT_ROOT/cub/{criterion}_k{keep}/seed{s}/` (예: `maskedkd_k0.3/seed1`). 같은 seed의 `cub/kd/seedN`과 짝을 이룬다.
- 학습 중간 체크포인트를 에포치 10, 30, 60에 `ckpt_e{epoch}.pt`로 저장한다 (student만, 모델 가중치만). 나중에 학습 진행에 따른 fidelity를 잴 때 쓴다.
- 로그(log.csv)에 열을 추가한다: 이번 에포치의 평균 `agree`(마스킹 teacher vs 같은 배치의 full teacher)는 매 step 계산하면 비싸니, 에포치마다 배치 1개만 full teacher를 추가로 돌려 기록한다.

**실행 (스케줄러에 `--stage 2` 추가)**
- 기준 2개 × 비율 3개 × seed 3개 = 18 run, 이후 각 run 평가(기존 evaluate.py: clean + corruption).
- 의존성: Stage 0의 `cub/teacher/seed0/DONE`과 `cub/lr_selection.json`만 필요. 없으면 실행을 거부한다.
- `--criteria`, `--keeps`, `--seeds` 인자로 일부만 돌릴 수 있게 한다.

**요약 (`stage0/summarize_stage2.py` → `results/stage2_summary.md`)**

| 기준 | 비율 | teacher GFLOPs/img | test acc (mean ± std) | Full KD 대비 | seed별 차이 |
|---|---|---|---|---|---|
| Full KD | 1.0 | | 81.73 ± … | 0 | |
| maskedkd | 0.5 / 0.3 / 0.15 | | | | |
| random | 0.5 / 0.3 / 0.15 | | | | |
| CE | – | 0 | 80.12 ± … | | |

- Full KD와 CE 값은 Stage 0 결과(`eval.json`)를 읽어 쓴다.
- **여유 구간 판정 (사전 고정):** maskedkd의 평균이 Full KD보다 0.5%p 이상 낮은 가장 큰 비율을 "MaskedKD 한계 비율"로 표시한다. 어떤 비율에서도 0.5%p 이상 떨어지지 않으면 "0.15에서도 손실 없음 → 더 낮은 비율(0.1, 0.05) 추가 필요"라고 쓴다.
- 같은 비율에서 maskedkd와 random의 차이도 seed별로 쓴다.

---

## 6. 검증 체크리스트

- [ ] 새 브랜치 `stage12`, Stage 0 브랜치·서버 폴더 무변경
- [ ] `teacher_forward` 기본값 동작 동일, keep_idx 전체·순서 섞기 테스트
- [ ] hook attention = 직접 계산 attention
- [ ] 크롭 박스를 반환하는 transform의 이미지 값이 기존과 비트 단위로 동일
- [ ] `random` 기준이 전역 RNG를 건드리지 않음: `kd` vs `maskedkd`/`random` 같은 seed 배치·mixup 해시 동일
- [ ] FLOPs가 토큰 수에 맞게 줄어듦 (keep 0.5에서 teacher GFLOPs가 대략 절반)
- [ ] Stage 0 unit test·smoke test가 그대로 통과
- [ ] 새 smoke test (`smoke_test_stage12.sh`): 가짜 데이터·random init·CPU로 캐시 생성 → fidelity(작은 N) → Stage 2 run 2개(1에포치) → 평가 → 요약까지 끝까지 돌아감. `/dev/shm`을 64MB로 줄여서도 통과
- [ ] smoke 통과 후 커밋·push하고, 서버 실행 명령을 README에 적는다 (아래 형식)

## 서버 실행 (README에 추가할 내용)
```bash
cd ~ && git clone -b stage12 https://github.com/jeehoo0507/aim-lab-test-5 aim-lab-stage12 && cd aim-lab-stage12
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
# Stage 0이 끝났는지 먼저 확인 (results/stage0_summary.md 존재)
setsid nohup bash -c 'uv run python -m stage0.make_attribution_cache --dataset cub && uv run python -m stage0.fidelity --dataset cub && ./run_stage0.sh --stage 2 --datasets cub' > $HOME/stage12.log 2>&1 < /dev/null &
```
- Stage 1(캐시 + fidelity)은 30분 안쪽, Stage 2(18 run)는 A5000 1장 기준 3시간 정도 예상.
- 같은 `OUTPUT_ROOT`를 쓰지만 Stage 0 폴더는 읽기만 한다.
