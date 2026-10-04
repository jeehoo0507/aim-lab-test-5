# Stage 3b: TAM이 rollout을 못 이긴 원인을 고친 재시도 (토큰 단위 증류 + 판단 근거 기반 teacher 신호)

## 배경 (Stage 3 파일럿 결과, seed 3개, 같은 seed 짝 차이)
- Waterbirds keep 0.15: MaskedKD WGA −18.5 (vs Full KD), rollout −2.1, tam −8.2, tam_r −7.3. tam_r − rollout: clean −0.44, **WGA −5.1**, corruption +1.1.
- CUB keep 0.15 (seed 2개): rollout > tam > MaskedKD. tam_oracle(teacher attention 직접 계산)이 가장 나쁨.
- 진단:
  1. **gap 정보가 student에게 전달될 통로가 없다.** student는 항상 전체 이미지를 보고, 받는 신호는 이미지당 logit 하나뿐이다. 토큰 선택은 "teacher가 무엇을 근거로 답하는가"만 바꾼다.
  2. **teacher 신호(마지막 블록 CLS attention)는 판단 근거가 아니다.** 배경의 고노름 토큰(register/sink)에 몰리고, teacher 자신의 지름길(배경)도 담는다. gap = "teacher는 보고 student는 덜 보는 곳"에 배경이 섞여 WGA를 깎는다.

## 목표
위 두 문제를 각각 고친 변형으로 **한 번만** 재시도한다. 결과를 보기 전에 판정 규칙을 고정한다 (7절).

## 작업 규칙
- `stage3-pilot`에서 새 브랜치 `stage3b`. 기존 기준(maskedkd, random, rollout, tam, tam_var, tam_oracle, tam_r)의 선택·학습 경로는 비트 단위로 그대로 (기존 테스트 + smoke 5단계 유지).
- 기존 인프라(스케줄러 `--stage 2`, 평가, `summarize_stage2`, `scripts/compare_runs.py`) 재사용. 새 기준은 `TRAIN_CRITERIA`에 추가만 한다.
- 짝맞춤 유지: 새 기준도 전역 RNG 사용 금지. 같은 seed에서 입력 배치·mixup 해시가 kd와 같아야 한다.

---

## 1. 판단 근거 기반 teacher 신호: `relevance` 캐시 (문제 2)

`make_attribution_cache`에 kind `relevance` 추가 (기존 `attn_last`, `rollout` 캐시는 그대로).
- 정의: **정답 클래스 logit에 대한 gradient로 가중한 attention rollout** (Chefer et al. 2021의 "gradient-weighted rollout" 단순화):
  - 각 블록 l의 attention A_l (B, H, N, N)과 그 gradient ∂y_c/∂A_l을 구한다 (y_c = 정답 클래스 logit; train 캐시이므로 라벨 사용 가능).
  - Ā_l = mean_h( relu(∇A_l ⊙ A_l) ), R ← R + Ā_l · R (R 초기값 I), 마지막에 CLS 행의 patch 부분 (196).
- **고노름 토큰 제외:** 마지막 블록 입력의 patch 토큰 L2 노름이 이미지 내 중앙값의 `--sink-ratio`(기본 3.0)배를 넘는 토큰은 relevance를 0으로 둔다 (register/sink 토큰). 제외 비율을 manifest에 기록.
- teacher는 fused SDPA라 attention이 밖으로 안 나온다. 기존 `attention.py`의 재계산 방식으로 grad가 흐르는 attention 행렬을 만든다 (hook에서 `torch.no_grad()` 쓰지 말 것; 캐시 생성 전용 함수로 분리).
- 오프라인 1회 계산 (train 전체, 이미지당 teacher forward+backward 1회). 비용은 manifest에 기록.
- `fidelity`에 `teacher_cache:relevance`, `teacher_oracle:relevance` 추가 (oracle은 그 view에서 직접 계산; test view는 정답 라벨로 계산하되 표에 "라벨 사용" 표시).
- 진단: relevance 상위 k 토큰 중 고노름 토큰 비율, attn_last 대비. CUB는 bounding box(`bounding_boxes.txt`)가 있으면 **상위 k 토큰 중 새 박스 안 비율**을 attn_last / rollout(student) / relevance로 비교해 `results/stage3b_attr_diag.md`에 기록.

## 2. 토큰 단위 증류 (문제 1)

학습 loss에 남긴 토큰 위치의 특징 증류 항을 추가한다. teacher는 이미 남긴 토큰만 계산하므로 **teacher 비용은 그대로**다.
- teacher: `teacher_forward(teacher, x, keep_idx)`가 마지막 블록(norm 적용 후) patch 특징 h_t (B, k, 768)도 반환하도록 옵션 추가.
- student: 같은 forward에서 마지막 블록 patch 특징 h_s (B, 196, 192; DeiT-Tiny) → keep_idx 위치만 gather → 학습용 선형 사상 P: 192→768 (student와 함께 학습, 평가에는 안 씀).
- L_tok = Σ_i w_i · (1 − cos(P(h_s,i), h_t,i)) / Σ_i w_i, i ∈ 남긴 토큰.
  - w_i = 1 (shared 또는 일반 선택), **gap 토큰은 w_gap** (기본 2.0).
- 전체 loss = Stage0Loss(CE + KD, α=0.5, τ=1) + β · L_tok, 기본 **β = 1.0**. 인자 `--tok-beta`, `--tok-gap-weight`.
- mixup/cutmix: h_t는 섞인 입력에서 계산되므로 student 특징과 위치가 맞다 (추가 처리 불필요).
- log.csv에 `train_tok` (L_tok 평균) 추가. config에 `tok_beta`, `tok_gap_weight`, `tam_kind`.

## 3. 새 기준

| 기준 | 토큰 선택 | teacher 신호 | 토큰 증류 | 확인하려는 것 |
|---|---|---|---|---|
| `rollout_f` | rollout 상위 k | – | 균일 (w=1) | 토큰 증류 자체의 효과 |
| `tam_rf` | tam_r (풀 + gap) | attn_last 캐시 | gap 가중 | 통로를 만들면 gap이 도움이 되는가 |
| `tam_rv` | tam_r | **relevance 캐시** | 없음 | teacher 신호만 고치면 되는가 |
| `tam_rvf` | tam_r | **relevance 캐시** | gap 가중 | 둘 다 고친 최종 형태 |

모두 tam_r과 같은 gap 커리큘럼(0.1→0.5)과 후보 풀(2k). 선택 비용은 rollout과 같다.

## 4. 실행 계획 (두 서버, 각 서버 안에서 짝 비교)
- 두 서버의 Stage 0 결과는 같은 teacher·기준 run을 재현한다 (Waterbirds teacher/KD/CE가 소수점 둘째 자리까지 일치). 그래도 짝 비교는 같은 서버 안의 run끼리만 한다.
- **cv2 (Waterbirds):** relevance 캐시 → fidelity → `rollout_f tam_rf tam_rv tam_rvf` × keep 0.15 × seed 0 1 2 (12 run). 기준 run(rollout, tam_r, maskedkd, kd)은 cv2에 이미 있다.
- **cv (CUB):** relevance 캐시 → fidelity·attr 진단 → 같은 4개 × keep 0.15 × seed 0 1 2 (12 run) + 비교용 `rollout tam_r` seed 2 및 `tam_r` seed 0 1 (cv에 없는 것만, 스케줄러가 알아서 건너뜀).

## 5. 요약
`summarize_stage2`에 새 기준 행, 그리고 `scripts/compare_runs.py`로 아래 짝 차이를 `results/stage3b_summary.md`에 기록:
- vs rollout: rollout_f, tam_rf, tam_rv, tam_rvf (Waterbirds는 WGA·clean·corruption, CUB는 clean·corruption)
- tam_rf − rollout_f (gap 가중의 순수 효과), tam_rvf − tam_rf (teacher 신호 교체의 효과)

## 6. 검증 체크리스트
- [ ] 기존 기준 선택·학습 비트 동일 (smoke 5단계, 기존 unit test)
- [ ] relevance: 정답 라벨을 바꾸면 맵이 바뀐다; 고노름 토큰은 0; 합성 입력에서 grad 경로가 살아 있다 (requires_grad 확인); oracle relevance가 같은 view에서 캐시 변환과 동일 계산을 쓴다
- [ ] 토큰 증류: β=0이면 tam_r / rollout과 최종 가중치 비트 동일; teacher 비용(FLOPs) 불변; P는 평가 checkpoint에서 제외되거나 무시됨
- [ ] 짝맞춤: kd / rollout / rollout_f / tam_rf / tam_rv / tam_rvf 입력 배치·mixup 해시 동일
- [ ] smoke (`smoke_test_stage3b.sh`, 가짜 데이터·CPU·/dev/shm 64MB): relevance 캐시 → fidelity → 4개 기준 1에포치 → 평가 → 요약
- [ ] smoke 통과 후 커밋·push, README에 두 서버 실행 명령 추가

## 7. 판정 (사전 고정, 결과 보기 전)
"이긴다" = 같은 seed 3개 모두 높고 평균 차이가 아래 이상.
- **TAM 유지:** `tam_rvf` 또는 `tam_rf`가 rollout을 **Waterbirds keep 0.15 WGA에서 +1.0%p 이상** 이기고, clean에서 −0.3%p보다 나쁘지 않으며, CUB keep 0.15 clean에서도 rollout보다 나쁘지 않다(평균 ≥ −0.2%p).
- **토큰 증류만 유효:** `rollout_f`가 rollout을 이기고 TAM 변형은 `rollout_f`를 못 이김 → 기여는 "토큰 증류 + rollout 선택"으로 정리, teacher 신호 기반 선택은 접는다.
- **둘 다 아님:** TAM 접고, 논문 축을 "토큰 축소 KD의 지름길 편향 증폭(MaskedKD WGA −18.5) + student rollout 선택으로 완화"로 정리한다.
