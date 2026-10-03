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
