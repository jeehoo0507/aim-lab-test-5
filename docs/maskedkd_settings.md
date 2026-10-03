# MaskedKD 설정 확인 (Stage 0, 0절)

출처: https://github.com/effl-lab/MaskedKD, commit `96d052da7346441e2425876a9ad37ff8c87fe383` (2024-11-07).
아래 라인 번호는 모두 이 commit 기준이다.

## 결론 (먼저)

**README 명령 + `main.py` 기본값으로 실제 적용되는 train 증강은 RRC + horizontal flip만이 아니다.**
기본값으로 RandAugment, random erasing, mixup, cutmix, label smoothing 0.1, repeated augmentation이
모두 켜져 있다. Stage 0 지침에 따라 **코드 작성 전에 멈추고 팀 결정을 기다린다.**

| 증강/정규화 | 기본값 | 위치 | 실제 적용 여부 |
|---|---|---|---|
| RandomResizedCrop | scale=(0.08, 1.0), ratio=(3/4, 4/3), bicubic | `transforms_factory.py:81-84`, `main.py:101` | 적용 |
| Horizontal flip | p=0.5 | `transforms_factory.py:60,85-86` | 적용 |
| RandAugment (`--aa`) | `rand-m9-mstd0.5-inc1` | `main.py:97` → `datasets.py:87` → `transforms_factory.py:103-104` | **적용** |
| Color jitter | 0.3 | `main.py:95` | 미적용 (`--aa`가 켜져 있으면 `elif`로 건너뜀, `transforms_factory.py:110`) |
| Random erasing (`--reprob`) | 0.25, mode=pixel, count=1 | `main.py:115-120` → `datasets.py:89-91` → `transforms_factory.py:132-134` | **적용** |
| Mixup | alpha 0.8 | `main.py:125`, 생성 `main.py:229-234`, 적용 `engine.py:38-39` | **적용** |
| Cutmix | alpha 1.0, switch prob 0.5, mixup-prob 1.0, mode=batch | `main.py:127-136` | **적용** |
| Label smoothing (`--smoothing`) | 0.1 | `main.py:100` | **적용** (mixup 타깃에 섞여 들어감, `main.py:234`) |
| Repeated augmentation | True (3회 반복) | `main.py:104-106`, `main.py:191-194`, `samplers.py:16,48` | **적용** |
| 3-Augment (`augment.py`) | – | `augment.py:90` `new_data_aug_generator` | 미사용 (어디서도 import/호출 안 됨) |

## 1. KD 손실 (`losses.py`)

```python
# losses.py:32-49
len_keep = torch.topk(attn.mean(dim=1)[:,0,1:], self.len_num_keep).indices
with torch.no_grad():
    teacher_outputs = self.teacher_model(inputs, len_keep, self.maskedkd)
base_loss = self.base_criterion(outputs, labels)
if self.distillation_type == 'soft':
    T = self.tau
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
```

- soft KD = `KL(p_T^τ || p_S^τ)`, `KLDivLoss(reduction='batchmean')` → **배치 크기로 나눔** (클래스 차원 합, 배치 평균). (`losses.py:43`)
- **τ² 곱함** (`* (T * T)`, `losses.py:44`). τ=1이면 영향 없음.
- 결합: `L = (1-α)·base + α·KD` (`losses.py:49`).
- `base_criterion`은 기본 설정에서 mixup이 켜져 있으므로 `SoftTargetCrossEntropy` (`main.py:262-264`). mixup을 끄면 smoothing>0일 때 `LabelSmoothingCrossEntropy(0.1)` (`main.py:265-266`), 둘 다 0일 때만 `CrossEntropyLoss` (`main.py:267-268`).
- teacher는 student와 **같은 (mixup 적용 후) 입력 텐서**를 받는다 (`engine.py:39,43`, `losses.py:36`). no_grad (`losses.py:35`), `teacher_model.eval()` (`main.py:292`).
- teacher는 cls token 출력만 head에 넣는다 (`models_teacher.py:274,276-279`). 토큰 선택은 pos_embed를 더한 뒤 gather (`models_teacher.py:261-266`). student attention은 마지막 블록 attention의 head 평균, cls→patch 행 (`models_student.py:259-266`, `losses.py:32`).
- AMP autocast 안에서 계산 (`engine.py:41-43`).

## 2. README 명령 기준 값

```
--model deit_small_patch16_224 --teacher_model deit_base --epochs 300 --batch-size 128
--distillation-type soft --distillation-alpha 0.5 --distillation-tau 1 --input-size 224
--maskedkd --len_num_keep 98   (8 GPU, torch.distributed.launch)
```

- distillation-type soft, α=0.5, τ=1 (`main.py:34-36` 기본값과도 동일)
- batch-size 128 **GPU당** (8 GPU → 총 1024). argparse 기본값은 64 (`main.py:38`).
- 증강 관련 인자 없음 → 아래 argparse 기본값 사용.

## 3. `main.py` argparse 기본값

| 인자 | 기본값 | 라인 |
|---|---|---|
| `--smoothing` | 0.1 | 100 |
| `--mixup` | 0.8 | 125 |
| `--cutmix` | 1.0 | 127 |
| `--mixup-prob` / `--mixup-switch-prob` / `--mixup-mode` | 1.0 / 0.5 / batch | 131-136 |
| `--aa` | `rand-m9-mstd0.5-inc1` | 97 |
| `--color-jitter` | 0.3 (AA 켜지면 미사용) | 95 |
| `--reprob` / `--remode` / `--recount` | 0.25 / pixel / 1 | 115-120 |
| `--repeated-aug` | True | 104-106 |
| `--train-interpolation` | bicubic | 101 |
| RRC scale | 인자 없음 → (0.08, 1.0) | `datasets.py:83-92`에서 scale 미전달, `transforms_factory.py:81` |
| `--opt` | adamw, eps 1e-8, betas 기본 | 55-60 |
| `--weight-decay` | 0.05 | 65 |
| `--lr` | 5e-4, **선형 스케일** `lr × batch × world_size / 512` (`--unscale-lr` 없으면) | 70, 252-254 |
| `--sched` | cosine | 68 |
| `--warmup-epochs` / `--warmup-lr` | 5 / 1e-6 | 85, 78 |
| `--min-lr` | 1e-5 | 80 |
| `--cooldown-epochs` | 10 (timm cosine은 epochs+cooldown 만큼 스케줄) | 87 |
| `--clip-grad` | None | 61 |
| `--drop-path` | 0.1 | 51 |
| `--drop` | 0.0 | 49 |
| `--epochs` | 300 | 39 |
| `--eval-crop-ratio` | 0.875 | 156 |
| `--train-mode` | True (student `model.train()`) | 108-110, `engine.py:26` |

참고: README 설정(8 GPU × 128)에서 실제 lr = 5e-4 × 1024 / 512 = **1e-3**.

## 4. 실제 transform

### Train (`datasets.py:79-98` → `transforms_factory.py:56-139`)
1. `RandomResizedCropAndInterpolation(224, scale=(0.08,1.0), ratio=(3/4,4/3), interpolation=bicubic)`
2. `RandomHorizontalFlip(p=0.5)`
3. `rand_augment_transform('rand-m9-mstd0.5-inc1', translate_const=100, img_mean=ImageNet mean)`
4. `ToTensor()` → `Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)`
5. `RandomErasing(p=0.25, mode='pixel', max_count=1)`
6. (배치 단위, GPU) `Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1)` (`engine.py:38-39`)
7. 샘플러: `RASampler(num_repeats=3)` — 각 이미지를 3번 반복해서 rank에 나눔, 에포치당 `len//256*256/world_size`개만 사용 (`samplers.py:35,58`)

### Test (`datasets.py:100-110`)
1. `Resize(int(224/0.875)=256, interpolation=3 (bicubic))`
2. `CenterCrop(224)`
3. `ToTensor()` → `Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)`

## 5. Stage 0 고정 설정과의 차이 (팀 결정 필요)

| 항목 | Stage 0 계획 | MaskedKD 실제 기본값 |
|---|---|---|
| Train 증강 | RRC + flip | RRC(0.08–1.0, bicubic) + flip + RandAugment + RandomErasing 0.25 + Mixup 0.8/Cutmix 1.0 + Repeated aug ×3 |
| Label smoothing | 0.0 (assert) | 0.1 |
| Base criterion | CE | SoftTargetCrossEntropy (mixup) |
| Student 초기화 | ImageNet pretrained | scratch (`main.py:238-242`, pretrained 미지정) |
| Teacher | 타깃 데이터셋 fine-tune | ImageNet pretrained 그대로 (`main.py:275-276`) |
| lr | {5e-5, 1e-4, 3e-4} 탐색 | 5e-4 × batch/512 |
| drop path | 미지정 | 0.1 |
| warmup lr / min lr / cooldown | 미지정 | 1e-6 / 1e-5 / 10 에포치 |
| Test Resize 보간 | 미지정 | bicubic |

KD 손실 식(KL batchmean × τ², α=0.5, τ=1), AdamW, wd 0.05, cosine, warmup 5, test Resize(256)→CenterCrop(224),
ImageNet mean/std는 Stage 0 계획과 일치한다.

## 6. Stage 0 채택 설정 (팀 결정, 2026-10-03)

구현 위치: `stage0/common.py` (상수), `stage0/datasets.py` (transform), `stage0/losses.py`, `stage0/train.py`.
"MaskedKD와 다름"인 항목마다 이유를 적었다.

| 항목 | Stage 0 | MaskedKD | 다름 | 이유 |
|---|---|---|---|---|
| Teacher | `deit_base_patch16_224` ImageNet-1k → 타깃 데이터셋 fine-tune | ImageNet pretrained 그대로 | 다름 | 타깃 데이터셋(CUB/Waterbirds)은 ImageNet과 클래스가 달라 fine-tune 없이는 teacher logit을 쓸 수 없음 |
| Student 초기화 | `deit_tiny_patch16_224` ImageNet-1k pretrained (head 새로 초기화) | scratch | 다름 | 소규모 데이터에서 scratch student는 KD/CE 모두 성능이 낮고, 이전 실험의 실패 요인 제거를 위해 teacher·student 모두 pretrained로 통일 |
| RRC | scale (0.08, 1.0), ratio (3/4, 4/3), bicubic | 동일 | 같음 | |
| Horizontal flip | p=0.5 | 동일 | 같음 | |
| RandAugment | `rand-m9-mstd0.5-inc1` (teacher, student 모두) | 동일 | 같음 | |
| Color jitter | 미적용 (RandAugment 사용 시 꺼짐) | 동일 | 같음 | |
| Random erasing | 0.25, pixel, count 1 (teacher, student 모두) | 동일 | 같음 | |
| Mixup / Cutmix (student) | 0.8 / 1.0, prob 1.0, switch 0.5, batch mode | 동일 | 같음 | |
| Mixup / Cutmix (teacher) | 끔 (assert) | (teacher 학습 없음) | 다름 | teacher fine-tune에서 soft target 학습을 피하고 hard label로 학습 (이전 실패: teacher 출력이 label smoothing에 의해 평탄화) |
| Label smoothing (student) | 0.1 (mixup 타깃에 포함) | 0.1 | 같음 | |
| Label smoothing (teacher) | 0.0 (assert) | (teacher 학습 없음) | 다름 | 이전 실험의 KD≈CE 원인. teacher 출력의 오답 클래스 정보가 smoothing으로 균일해지는 것을 막음 |
| Repeated aug | 끔 (모든 run) | 켬 (×3) | 다름 | RASampler는 에포치당 고유 이미지 수를 1/3로 줄임 (`samplers.py:35,58`); 수천 장 규모 데이터에서는 손해 |
| CE 기준 손실 (student) | SoftTargetCrossEntropy (mixup 타깃) | 동일 | 같음 | |
| CE 기준 손실 (teacher) | CrossEntropy(label_smoothing=0) | – | – | |
| KD 손실 | `(1-α)·CE + α·KLDiv(batchmean)(log_softmax(s/τ), softmax(t/τ))·τ²`, α=0.5, τ=1 | 동일 (`losses.py:43-49`) | 같음 | |
| KD teacher 입력 | student와 같은 mixup 적용 후 텐서, `eval()` + `no_grad` | 동일 (`engine.py:39-43`) | 같음 | |
| Student CE vs KD | 완전히 같은 레시피 (같은 seed → 같은 head 초기화·데이터 순서·증강·mixup 난수) | – | – | 비교의 유일한 차이를 KD 항으로 한정 |
| Optimizer | AdamW, wd 0.05, eps 1e-8, bias/norm/pos_embed/cls_token wd 제외 | 동일 (timm create_optimizer) | 같음 | |
| lr | sweep: teacher {5e-5, 1e-4}, student CE {5e-5, 1e-4, 3e-4}; CE에서 고른 lr을 KD에도 사용 | 5e-4 × batch/512 (선형 스케일) | 다름 | fine-tune 규모(소규모 데이터, pretrained)에서 ImageNet scratch용 lr은 맞지 않음. 데이터셋별로 val로 선택 |
| Schedule | cosine, warmup 5 에포치 (warmup lr 1e-6), 에포치 단위 갱신 | 동일 | 같음 | |
| min lr | base lr / 100 | 1e-5 (절대값) | 다름 | lr이 5e-5~3e-4라 절대값 1e-5는 lr마다 감쇠 비율이 달라짐 → 비율로 통일 |
| Cooldown | 0 (총 100 에포치 고정) | 10 | 다름 | 모든 run의 에포치 수를 100으로 동일하게 유지 |
| Epochs | 100 | 300 | 다름 | 소규모 데이터 fine-tune에 충분, 계산량 절약 |
| Batch | 128 (유효), OOM 시 gradient accumulation | GPU당 128 × 8 GPU | 다름 | 단일 GPU run. mixup은 128 전체 배치에 적용한 뒤 micro-batch로 나눠서 micro-batch 크기와 무관하게 동일 |
| drop_path | 0.1 (teacher fine-tune, student 모두) | 0.1 | 같음 | |
| Train interpolation | bicubic | 동일 | 같음 | |
| Test transform | Resize(256, bicubic) → CenterCrop(224) → Normalize(ImageNet) | 동일 | 같음 | |
| AMP | fp16 autocast + GradScaler | 동일 | 같음 | |
| Student 평가 | 마지막 에포치 | best val | 다름 | val 선택에 의한 낙관적 편향 제거 (사전 고정 규칙) |
| Teacher 체크포인트 | val 최고 에포치 | – | – | |
| Teacher 진단 view | train 이미지 + train 증강 (RRC + flip + RandAugment + erasing), **mixup 전**, seed 0, 1 pass | – | – | KD 때 teacher가 실제로 보는 증강 분포에서 teacher 출력의 정보량을 측정. mixup은 라벨 정의를 바꾸므로 제외 |

구현상 결정 (팀 결정 범위 밖, 결과에 영향 없음):
- lr 선택 run (seed 0)은 본 run seed 0과 설정이 완전히 같으므로, 선택된 lr의 sweep run을 `outputs/{dataset}/{teacher|ce}/seed0/`로 승격(hard link)해서 다시 학습하지 않는다. 데이터셋마다 DeiT-B teacher 1회, CE 1회를 절약한다.
- student lr 선택 기준은 CE의 **마지막 에포치** val acc(student 평가 규칙과 일치), teacher는 **best 에포치** val acc(teacher 체크포인트 규칙과 일치). 동점이면 작은 lr.
- Waterbirds는 C=2라 "오답 클래스 정규화 엔트로피 / log(C−1)"이 정의되지 않아 `null`로 기록한다.
- `imagecorruptions` 1.1.2는 최신 scikit-image/NumPy와 호환되지 않아 `stage0/corruption_compat.py`에서 인자 이름만 맞춰준다 (`multichannel` → `channel_axis`, `np.float_` → `float64`). impulse_noise의 난수를 numpy seed에 연결해 캐시를 재현 가능하게 했다. corruption 내용 자체는 바뀌지 않는다.
