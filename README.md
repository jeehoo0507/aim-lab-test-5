# Stage 0: KD > CE 환경 확보

DeiT-B (ImageNet-1k → 타깃 데이터셋 fine-tune) teacher, DeiT-Ti (ImageNet-1k pretrained) student로
CUB-200-2011과 Waterbirds에서 CE vs Full KD를 비교한다. 마스킹은 없다. teacher forward는
`stage0/models.py:teacher_forward()` 하나로 분리되어 있어 이후 단계에서 토큰 선택을 끼울 수 있다.

설정 근거: [`docs/maskedkd_settings.md`](docs/maskedkd_settings.md). MaskedKD 코드 확인 결과와
하단의 "Stage 0 채택 설정" 표 참고.

## 서버 실행 순서

```bash
# 0) 환경 (uv). torch 2.9.1 = CUDA 12.8 빌드 (Blackwell sm_120 포함, NVIDIA driver >= 570 필요)
uv sync
export DATA_ROOT=/data/stage0 OUTPUT_ROOT=/data/stage0_outputs HF_HOME=/data/hf GPUS="0,1"

# 1) 사전 점검: GPU/아키텍처/디스크/권한/인터넷/메모리 → $OUTPUT_ROOT/preflight.json
uv run python -m stage0.preflight

# 2) 데이터 다운로드 + split 고정 (크기가 공식 값과 다르면 에러)
uv run python -m stage0.prepare_data

# 3) smoke test (실제 데이터·가중치, 64장·1에포치, 전 과정 + 병렬 + 재개 + 가드 테스트)
./smoke_test.sh

# 4) 전체 실행 (사용자 확인 후). tmux 안에서:
tmux new -s stage0 './run_stage0.sh'
./status.sh            # 진행 상황 표
```

### tmux 없이 한 번에 (1→4, 실패하면 그 단계에서 멈춤, 터미널을 닫아도 계속 돈다)

```bash
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
setsid nohup bash -c 'uv run python -m stage0.preflight && uv run python -m stage0.prepare_data \
  && ./smoke_test.sh && ./run_stage0.sh' > $HOME/stage0.log 2>&1 < /dev/null &
tail -f $HOME/stage0.log     # Ctrl+C로 보기만 멈춤 (실행은 계속)
```

컨테이너(Kubernetes, Coder 등)에서는 `os.cpu_count()`와 `free`가 호스트 전체 값을 보여준다. CPU 수와 RAM은
cgroup 제한을 읽어서 쓴다 (`stage0/common.py: cpu_count(), mem_limit_gb()`). preflight의 `RUNS_PER_GPU`
추천값은 GPU 메모리·CPU·RAM 중 가장 작은 쪽으로 정해진다. 감지가 틀리면 `STAGE0_CPUS`, `STAGE0_MEM_GB`로 덮어쓴다.

컨테이너에서 겪은 문제와 대응:
- `/dev/shm`이 64MB뿐이라 DataLoader worker가 `Bus error`로 죽음 → worker가 배치를 numpy로 만들어 파이프로 넘긴다
  (`stage0/engine.py: numpy_collate`). `/dev/shm` 크기와 무관하고 값은 동일하다.
- `ImportError: libGL.so.1` (imagecorruptions가 GUI용 opencv-python을 끌어옴) → `pyproject.toml`에서 opencv-python을
  제외하고 headless만 설치한다. 예전 환경을 갱신할 때는 `rm -rf .venv && uv sync` (두 패키지가 같은 `cv2/` 파일을
  공유해서 제자리 제거가 깨질 수 있음).

인터넷이 막힌 서버: preflight가 필요한 파일 목록을 출력한다. 요약하면
- timm 가중치 `timm/deit_base_patch16_224.fb_in1k`, `timm/deit_tiny_patch16_224.fb_in1k`를
  다른 머신에서 받아 `$HF_HOME/hub/`에 복사
- `CUB_200_2011.tgz`, `waterbird_complete95_forest2water2.tar.gz`를 `$DATA_ROOT/downloads/`에 두고
  `prepare_data.py` 실행

## run_stage0.sh가 하는 일

| 단계 | 내용 | 의존성 |
|---|---|---|
| teacher lr sweep | lr ∈ {5e-5, 1e-4}, seed 0, 데이터셋별 | 없음 (즉시 시작) |
| student lr sweep | CE, lr ∈ {5e-5, 1e-4, 3e-4}, seed 0 | 없음 (즉시 시작) |
| corruption 캐시 | 1,000장 × 15 × 5, PNG, CPU | 없음 (즉시 시작) |
| lr 선택 | `results/lr_selection.md`; 선택된 sweep run을 seed 0 본 run으로 승격 | sweep |
| teacher 진단 | `outputs/{ds}/teacher/seed0/diagnosis.json` | teacher lr 선택 |
| CE seed 1, 2 / KD seed 0, 1, 2 | 선택된 student lr | student lr 선택 (+ teacher) |
| 평가 | 모든 run의 `eval.json` | 학습 + 캐시 |
| 요약 | `results/stage0_summary.md` (표 + 판정) | 전부 |

- `GPUS`에 적은 GPU만 쓴다 (run마다 `CUDA_VISIBLE_DEVICES` 1개). `RUNS_PER_GPU` 기본값은 preflight 측정값.
  student run은 한 GPU에 여러 개 겹쳐 돌고, DeiT-B teacher fine-tune은 GPU당 최대 1개이며 측정한
  메모리 비율만큼 슬롯을 차지한다 (`TEACHER_SLOTS`로 변경 가능).
- 같은 명령을 다시 치면 남은 것만 돈다 (`DONE`이 있는 run은 건너뛰고, 중단된 run은 `last.pt`에서 이어서).
- 실패한 run은 `FAILED`(traceback)을 남기고, 다른 run은 계속된다. 실패한 run에 의존하는 run은 BLOCKED.
- α=1.0 fallback (5절, 사용자 확인 후): `./run_stage0.sh --fallback-alpha1`

## 출력 구조

```
$OUTPUT_ROOT/
  preflight.json
  {cub,waterbirds}/
    lrsel/{teacher,ce}_lr{lr}/seed0/     lr sweep runs
    teacher/seed0/   best.pt last.pt log.csv config.json train.log DONE diagnosis.json eval.json
    ce/seed{0,1,2}/  last.pt log.csv config.json train.log DONE eval.json
    kd/seed{0,1,2}/  (같음, log.csv에 train_kd 열)
    lr_selection.json
  _scheduler/state.json, logs/
$DATA_ROOT/corruptions/{cub,waterbirds}/   (CORRUPTION_ROOT로 변경 가능)
results/lr_selection.md, results/stage0_summary.md
```

## 개별 실행

```bash
uv run python -m stage0.train --mode teacher --dataset cub --seed 0 --lr 1e-4
uv run python -m stage0.train --mode kd --dataset cub --seed 1 --lr 1e-4     # teacher/seed0/best.pt 필요
uv run python -m stage0.diagnose_teacher --dataset cub
uv run python -m stage0.make_corruptions --workers 16
uv run python -m stage0.evaluate --dataset waterbirds --mode kd --seed 0
uv run python -m stage0.summarize
```

---

# Stage 1 + 2: teacher 토큰 마스킹, fidelity, MaskedKD baseline (브랜치 `stage12`)

Stage 0 폴더(`~/aim-lab-test-5`, 브랜치 `claude/upbeat-curie-lhrxvg`)는 Stage 0 스케줄러가 실행 중에 코드를 import하므로
건드리지 않는다. Stage 1+2는 별도 폴더에 clone해서 돌린다. 같은 `OUTPUT_ROOT`를 쓰지만 Stage 0 run 폴더
(`cub/{teacher,ce,kd,lrsel}`, `cub/lr_selection.json`)는 읽기만 한다. 새 결과는 아래 경로에만 쓴다.

```bash
cd ~ && git clone -b stage12 https://github.com/jeehoo0507/aim-lab-test-5 aim-lab-stage12 && cd aim-lab-stage12
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
# Stage 0이 끝났는지 먼저 확인 (results/stage0_summary.md 존재, ~/aim-lab-test-5 기준)
ls ~/aim-lab-test-5/results/stage0_summary.md
setsid nohup bash -c 'uv run python -m stage0.make_attribution_cache --dataset cub && uv run python -m stage0.fidelity --dataset cub && ./run_stage0.sh --stage 2 --datasets cub' > $HOME/stage12.log 2>&1 < /dev/null &
tail -f $HOME/stage12.log            # Ctrl+C는 보기만 멈춤
OUTPUT_ROOT=$OUTPUT_ROOT ./status.sh --stage 2
```
- Stage 1(캐시 + fidelity)은 30분 안쪽, Stage 2(18 run + 평가)는 A5000 1장 기준 3시간 정도 예상.
- 같은 명령을 다시 치면 끝난 것은 건너뛴다 (캐시 manifest, run별 `DONE`/`eval.json`).
- 일부만: `./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd --keeps 0.3 --seeds 0 1`
- 학습 중간 체크포인트로 fidelity 다시 재기:
  `uv run python -m stage0.fidelity --dataset cub --student-ckpt $OUTPUT_ROOT/cub/maskedkd_k0.3/seed0/ckpt_e10.pt --out-name stage1_fidelity_mkd03_e10`
- 서버에서 smoke: `GPUS=0 DATA_ROOT=... ./smoke_test_stage12.sh` (toy-scale 출력은 `smoke/run12/` 아래에만 쓴다)

| 파일 | 내용 |
|---|---|
| `stage0/models.py: teacher_forward(teacher, images, keep_idx=None)` | `_pos_embed` 뒤에서 cls + 고른 patch 토큰만 남김 (MaskedKD와 같은 위치). `None`이면 Stage 0과 동일 |
| `stage0/attention.py` | attn 모듈 forward hook으로 attention 재계산 (fused SDPA라 밖으로 안 나옴). 필요할 때만 hook 등록 |
| `stage0/masking.py` | 기준: maskedkd, random(전용 generator), rollout, teacher_cache, teacher_oracle |
| `stage0/attribution.py`, `make_attribution_cache.py` | teacher attribution 캐시(`cub/attribution_cache/`, train split 전체, 원본 전체 224 리사이즈) + RRC 박스/flip 좌표 변환 |
| `stage0/fidelity.py` | Stage 1 → `results/stage1_fidelity.{csv,md,png}` |
| `stage0/train.py --mode maskedkd --mask-criterion {maskedkd,random} --keep K` | Stage 2 → `cub/{criterion}_k{keep}/seed{s}/` (+ `ckpt_e{10,30,60}.pt`, log.csv `mask_agree`) |
| `stage0/scheduler.py --stage 2` | 18 run + 평가 + `results/stage2_summary.md`. 상태는 `$OUTPUT_ROOT/_scheduler_stage2/` (Stage 0 상태와 분리) |
| `stage0/summarize_stage2.py` | 표 + 여유 구간 판정 (사전 고정: Full KD 대비 maskedkd 평균 −0.5%p 이상 하락하는 가장 큰 비율) |

attribution 캐시 → 크롭 변환의 한계: RandAugment의 기하 변환(회전·전단·평행이동), random erasing, mixup/cutmix는
반영하지 않는다. 14×14 원본 맵을 크롭 박스로 bilinear 리샘플링하므로 작은 크롭에서는 해상도가 낮다. 이 근사의 오차는
Stage 1에서 `teacher_cache` vs `teacher_oracle`(같은 view에서 teacher를 직접 돌린 attribution)의 fidelity 차이와
top-k 겹침 비율(`oracle_overlap`)로 잰다.

GFLOPs는 `FlopCounterMode` 실측값을 DeiT/MaskedKD 논문 관례(곱셈-덧셈 1회 = 1 FLOP, DeiT-B 17.6 G)로 보고한다
(카운터 값 ÷ 2). keep 0.5 → 8.70 / 17.56 = 0.50.

---

# Stage 3 pilot: TAM (Teachability-Aware Masking) vs MaskedKD — cv 서버 (브랜치 `stage3-pilot`)

TAM이 MaskedKD보다 나은 방향인지 빠르게 확인한다 (결론은 "가능성 있음 / 낮음"까지만). TAM이 지면 원인이 **아이디어**인지
**캐시의 부정확함**인지 가를 수 있게 `tam_oracle` 상한선 run과 캐시 진단을 함께 돌리고, 가장 강한 경쟁자인 student
`rollout`도 학습 baseline으로 넣는다. 비교는 **cv 서버 안에서만** (cv의 `kd/seed{0,1}`와 같은 teacher). cv2의 Stage 2 결과와
섞지 않는다 (teacher 체크포인트가 다름). Stage 0·2 폴더는 읽기만 한다.

```bash
cd ~ && git clone -b stage3-pilot https://github.com/jeehoo0507/aim-lab-test-5 aim-lab-stage3 && cd aim-lab-stage3
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
setsid nohup bash -c 'uv run python -m stage0.make_attribution_cache --dataset cub && uv run python -m stage0.diagnose_cache --dataset cub && ./run_stage0.sh --stage 2 --datasets cub --criteria maskedkd rollout tam tam_var --keeps 0.3 0.15 --seeds 0 1 && ./run_stage0.sh --stage 2 --datasets cub --criteria tam_oracle --keeps 0.15 --seeds 0 1' > $HOME/stage3_pilot.log 2>&1 < /dev/null &
tail -f $HOME/stage3_pilot.log
OUTPUT_ROOT=$OUTPUT_ROOT ./status.sh --stage 2
```
- 18 run (maskedkd·rollout·tam·tam_var × keep 0.3·0.15 × seed 0·1 + tam_oracle × 0.15 × seed 0·1) + 평가. A5000 1장(5 슬롯) 기준
  4시간 안팎. 이미 끝난 run(`DONE`)은 건너뛴다. 스케줄러는 `tam_oracle`을 가장 늦게 시작한다 (우선순위 최하).
- 요약(`results/stage2_summary.md`)은 호출마다 끝난 run 전체로 다시 만들어진다 (두 번째 호출 뒤에 18 run 전부 포함).
- 캐시가 다른 teacher로 만들어져 있으면 `make_attribution_cache`가 거부한다 (`--overwrite`로 다시 만든다).
- 서버 smoke: `GPUS=0 DATA_ROOT=... ./smoke_test_stage3_pilot.sh` (출력은 `smoke/run3/` 아래)

### 학습 기준 (`train.py --mode maskedkd --mask-criterion C --keep K` → `cub/{C}_k{K}/seed{s}/`)
| 기준 | teacher 신호 T | 토큰 수 | 선택 비용 (GFLOPs/img, 실측) |
|---|---|---|---|
| `rollout` | – (student rollout 상위 k, Stage 1 `rollout`과 같은 인덱스) | k | 0.435 (전 블록 attention + rollout) |
| `tam` | 캐시 → 현재 크롭·flip으로 변환 | k | 0.022 (S용 student hook) |
| `tam_var` | 캐시 → 현재 크롭·flip으로 변환 | 버킷 k(1−δ)/k/k(1+δ), 평균 k | 0.022 |
| `tam_oracle` | 현재 view(mixup 전)에서 teacher 전체 forward의 attn_last (진단용) | k | 17.93 (teacher forward 1회 추가) |

TAM 선택 (`stage0/masking.py: tam_select, tam_buckets, TamSelector`). S = student 마지막 블록 CLS→patch attention
(MaskedKD와 같은 신호, 학습 forward의 hook) — MaskedKD 대비 차이가 "teacher 신호를 섞은 효과"만 되도록.
1. teacher 후보 풀 = `T` 상위 `m = min(196, 2k)`개 (나머지는 pruned)
2. shared = 풀 안에서 `S` 상위 `round((1−g)k)`개
3. gap = 풀의 나머지 중 `T` 상위 `k − k_shared`개
4. `g(epoch) = g0 + (g1 − g0)·epoch/(epochs−1)`, 기본 0.1 → 0.5 (`--tam-gap g0 g1`). T 종류 기본 `attn_last` (`--tam-kind rollout`)

`tam_var`: 크롭된 T를 합 1로 정규화한 상위 k개 질량(집중도)으로 micro-batch를 3등분 → 집중도 높은 1/3은 `k(1−δ)`, 낮은 1/3은
`k(1+δ)`, 나머지(나머지 이미지 포함)는 `k` (δ=0.33; keep 0.3: 40/59/78, keep 0.15: 19/29/39; |mean_k − k| ≤ 1 assert). 패딩·mask
없이 버킷마다 teacher를 따로 돌려 logit을 원래 순서로 합친다 (`models.teacher_forward_buckets`). 버킷 가중 평균 teacher GFLOPs는
고정 k 대비 +0.1% 이내이며 에포치마다 log.csv `teacher_gflops`에 실측값을 남긴다.

log.csv 추가 열: `tam_gap_ratio`, `mean_k`, `bucket_k`, `teacher_gflops` (tam 계열), `cache_oracle_overlap` (`tam`·`tam_var`:
에포치마다 첫 micro-batch에서 같은 view의 teacher top-k와 캐시 top-k의 겹침). config 추가 키: `tam_gap`, `tam_kind`, `tam_budget`,
`tam_delta`, `tam_teacher_signal` (cache/oracle), `selection_gflops` (rollout·tam 계열). 기존 `maskedkd`/`random`의 config·선택·
학습 경로는 stage12와 비트 단위로 같다 (`tests/test_stage3.py`, smoke 5단계에서 stage12 worktree와 최종 가중치 비교).

**한계:** tam/tam_var의 T는 mixup/cutmix **전** 원본 view의 크롭 박스로 변환한 캐시 기준이다 (캐시는 원본 이미지에 대해서만
있음). mixup 상대 이미지·cutmix로 붙여진 영역, RandAugment 기하 변환, random erasing은 T에 반영되지 않는다 (S는 mixup 후 입력의
student attention이라 반영됨). tam_oracle도 mixup 전 view에서 teacher를 돌린다.

### 캐시 진단 (`stage0/diagnose_cache.py` → `results/stage3_cache_diag.{csv,md}`, 학습 없음)
train 이미지마다 RRC view 4개(전용 RNG, seed 0, flip 포함, RandAugment 없음)에서 같은 view의 teacher attn_last(oracle)와 비교:
`cache_vs_view`(tam이 쓰는 캐시 변환 맵), `whole_vs_view`(새로 계산한 전체 이미지 맵을 같은 크롭 변환 → 크롭 변환 자체의 한계;
cache_vs_view와의 차이 = 캐시 저장 오차), `view_vs_view`(같은 view 두 번, 100%여야 함). 크롭 면적 구간 [0.08,0.2) [0.2,0.4)
[0.4,0.7) [0.7,1.0]과 flip 여부별 top-k overlap(keep 0.3/0.15)과 Spearman. 결론 규칙(사전 고정, keep 0.3 cache_vs_view): 가장 큰
구간 ≥ 70%이고 (큰 − 작은) ≥ 15%p → 해상도 문제(다중 스케일 캐시로 개선 여지), 가장 큰 구간 < 70% → 문맥 의존(캐시 방식의 한계),
그 외 혼합. 개선 방법은 구현하지 않았다.

### 파일럿 요약과 판단 (`summarize_stage2.py`, 사전 고정)
표에 rollout·tam·tam_var·tam_oracle 행과 선택 비용 열(tam_oracle은 "진단용, 비용 비교 대상 아님"), 같은 seed 짝 차이
(`tam/tam_var/tam_oracle − maskedkd`, `tam/tam_var − rollout`), tam_var 실측 GFLOPs, 학습 중 `cache_oracle_overlap`.
"이긴다" = 짝 seed 모두(≥2) 높고 평균 +0.3%p 이상.

| tam 또는 tam_var vs maskedkd | tam_oracle vs maskedkd (0.15) | 판단 |
|---|---|---|
| 이김 | – | 가능성 있음 → 본 실험(3 seed). rollout도 이기는지 함께 보고 (못 이기면 "rollout 대비 우위 없음") |
| 못 이김 | 이김 | 아이디어는 유효, 캐시가 병목 → 캐시 진단 결과로 캐시 개선 후 재실험 |
| 못 이김 | 못 이김 | 가능성 낮음 → 선택 규칙(풀 크기, g 스케줄) 재검토 |

필요한 run의 짝 seed가 2개 미만이면 "미완료 — 판단 보류".
