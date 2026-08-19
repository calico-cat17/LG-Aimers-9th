# LG Aimers 9기 — JM R Residual 개선 모델

JOA의 `F-Regime075` 제출 모델을 anchor로 유지하면서, `game_type == "R"`인 행에만 작은 CatBoost residual correction을 추가한 실험입니다. 최종 학습과 제출 ZIP 생성은 **Google Colab에서 수행**했습니다.

## 공식 결과

| 모델 | Public BSS | 직전 대비 |
|---|---:|---:|
| JOA `260818_F_regime075.zip` | 1126.8664003703 | - |
| **JM R Residual Multi-seed × 0.05** | **1127.5514185677** | **+0.6850181974** |

기존 JOA 모델의 F 전문가를 보존하고 R 영역만 보정한 결과, 공식 리더보드에서 약 `+0.685`점 개선됐습니다.

## 모델 구성

```text
JOA F-Regime075 anchor probability
        │
        ├─ game_type == F → JOA 예측을 그대로 사용
        │
        └─ game_type == R
             ├─ time-safe 공통 피처
             ├─ Candidate4 ExtraTrees/Beta/raw 메타 피처
             ├─ CatBoost residual model: seed 17
             ├─ CatBoost residual model: seed 42
             └─ CatBoost residual model: seed 777
                         │
                         └─ 평균 correction × 0.05 → 최종 확률
```

Residual target은 `control_success - JOA anchor probability`입니다. 세 모델의 correction을 산술평균하고 `[-0.12, 0.12]`로 제한한 뒤 R 행에만 `0.05`배 적용합니다.

최종 식은 다음과 같습니다.

```text
F: p_final = p_JOA
R: p_final = clip(p_JOA + 0.05 × mean(correction_seed17, correction_seed42, correction_seed777))
```

## 로컬 검증

최종 제출 코드에서 실제 생성 가능한 116개 피처만 사용해 2024 forward OOF를 다시 평가했습니다.

| 항목 | 값 |
|---|---:|
| JOA anchor BSS | 897.6979 |
| R residual candidate BSS | 899.4424 |
| ΔBSS | +1.7446 |
| Pitcher-cluster bootstrap 95% CI | [-0.7023, 4.1747] |
| P(ΔBSS > 0) | 92.65% |
| Seeds | 17, 42, 777 |
| 최종 seed별 tree 수 | 15, 21, 15 |

CI 하한은 0보다 작아 로컬 기준으로는 탐색적 후보였지만, 세 seed가 같은 개선 방향을 보였고 correction scale이 작아 실제 제출을 진행했습니다. 공식 리더보드에서도 개선 방향이 재현됐습니다.

## Colab 재현

실제 최종 재학습과 제출 ZIP 생성에 사용한 실행 기록은 다음 노트북에 포함되어 있습니다.

- [`notebooks/LG_AIMERS_JOA_R_Residual_Final_Trainer_Colab.ipynb`](notebooks/LG_AIMERS_JOA_R_Residual_Final_Trainer_Colab.ipynb)

이 노트북은 정리된 예시가 아니라 **실제 Google Colab 실행본**이며, 실행 출력과 당시 적용한 패키징 보정 셀이 포함되어 있습니다.

Google Drive의 `MyDrive/LG_AIMERS/` 아래에 다음 파일이 필요합니다.

```text
data/train.csv
data/test.csv
data/sample_submission.csv
LG_AIMERS_JOA_R_Residual_Final_Trainer_Code_v1.zip
260818_F_regime075.zip
JOA_strict_F_regime_OOF_validated_full_gpu.zip
JOA_Candidate4_Strict_Blend_Results.zip
```

노트북을 Colab에 업로드한 뒤 위에서 아래 순서로 실행하면 다음 제출 파일을 생성합니다.

```text
MyDrive/LG_AIMERS/joa_r_residual_final/submit_JOA_R_residual_multiseed005.zip
```

실제 대회 제출 시 파일명은 `sub_JM_R_res_multiseed005.zip`으로 변경했습니다. 외부 ZIP 이름만 변경했으며 내부 모델과 추론 로직은 동일합니다.

## 독립 예측 원칙

- 검증 연도보다 과거인 데이터로 모델을 학습하는 forward OOF 방식을 사용했습니다.
- 평가 데이터의 정답이나 평가 데이터 전체 분포를 사용하지 않습니다.
- 테스트 행 간 rolling·누적 집계를 만들지 않습니다.
- 테스트의 각 행과 학습 단계에서 저장한 artifact만 사용해 독립적으로 예측합니다.
- F 예측은 JOA anchor와 동일하게 유지합니다.

## 이번 JM 브랜치의 범위

이번 브랜치에는 위 R residual 개선 실험을 설명하는 README와 실제 Colab 실행 노트북만 추가합니다. 다른 실험용 `.py`, `.ipynb`, 중간 OOF 및 설계 파일은 검토 없이 포함하지 않습니다.

