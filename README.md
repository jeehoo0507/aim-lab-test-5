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
