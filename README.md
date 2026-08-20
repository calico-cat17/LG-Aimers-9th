# LG Aimers 9기 — JM R Residual 개선 모델

JOA의 `F-Regime075` 제출 모델을 anchor로 유지하면서, `game_type == "R"`인 행에만 CatBoost residual correction을 추가한 실험입니다. 최종 학습·검증·제출 ZIP 생성은 **Google Colab에서 수행**했습니다.

## 공식 결과

| 모델 | Public BSS | 직전 대비 |
|---|---:|---:|
| JOA `260818_F_regime075.zip` | 1126.8664003703 | - |
| JM R Residual Multi-seed × 0.05 | 1127.5514185677 | +0.6850181974 |
| **JM R Residual Multi-seed × 0.075** | **1127.8851982941** | **+0.3337797264** |

최종 0.075 모델은 anchor 대비 `+1.0187979238`, 기존 0.05 제출 대비 `+0.3337797264` 개선됐습니다.

## 모델 구성

```text
JOA F-Regime075 anchor probability
        │
        ├─ game_type == F → anchor 예측 유지
        │
        └─ game_type == R
             ├─ time-safe 공통 피처
             ├─ Candidate4 ExtraTrees/Beta/raw 메타 피처
             ├─ CatBoost residual model: seed 17
             ├─ CatBoost residual model: seed 42
             └─ CatBoost residual model: seed 777
                         │
                         └─ 평균 correction × 0.075 → 최종 확률
```

Residual target은 `control_success - JOA anchor probability`입니다. 세 seed의 correction을 산술평균하고 안전 범위로 제한한 뒤 R 행에만 적용합니다.

```text
F: p_final = p_JOA
R: p_final = clip(p_JOA + 0.075 × mean(correction_seed17, correction_seed42, correction_seed777))
```

## 0.075 선택 근거

2024 forward OOF에서 correction scale을 `0.000`부터 `0.150`까지 `0.005` 간격으로 비교했습니다.

| Scale | 2024 BSS | Anchor 대비 ΔBSS | Bootstrap 95% CI | P(ΔBSS > 0) | 공식 점수 |
|---:|---:|---:|---:|---:|---:|
| 0.050 | 899.4424 | +1.7446 | [-0.6586, 4.1840] | 92.23% | 1127.5514 |
| **0.075** | **899.8907** | **+2.1928** | **[-1.4346, 5.8776]** | **88.60%** | **1127.8852** |
| 0.100 | 900.0562 | +2.3583 | [-2.5131, 7.2879] | 83.57% | 미제출 |

0.100은 로컬 점수가 더 높았지만 불확실성도 커졌습니다. 0.075는 0.05보다 correction을 강화하면서 0.10의 높은 분산을 피하는 절충 후보로 선택했고, 공식 리더보드에서도 개선 방향이 재현됐습니다.

## Colab 및 검증 자료

- 최종 residual 학습·패키징: `notebooks/LG_AIMERS_JOA_R_Residual_Final_Trainer_Colab.ipynb`
- correction scale 탐색: `notebooks/LG_AIMERS_JOA_R_Scale_Sweep_Colab.ipynb`
- scale별 검증 결과: `reports/r_scale_sweep/scale_curve_2024.csv`
- 후보 요약: `reports/r_scale_sweep/scale_candidate_summary.csv`
- 제출 패키지 manifest: `reports/r_scale_sweep/submission_manifest.csv`
- 공식 제출 이력: `evaluation/leaderboard_history.csv`

Colab 실행에는 비공개 대회 데이터와 JOA anchor·OOF·Candidate4 artifact가 별도로 필요합니다. 이 데이터와 중간 artifact는 Git에 포함하지 않습니다.

## 최종 제출물

현재 유지 제출물은 다음 파일입니다.

```text
final/sub_JM_R_res_scale075_RE.zip
```

`_RE` 파일은 대회 코드 제출 형식에 맞게 최상위 구조를 정리한 재패키징본이며 예측 로직은 scale 0.075 제출과 동일합니다. 대용량 파일은 Git LFS로 관리합니다.

```text
SHA-256: 46aa2a15130ed9ba8d302f203fa42c8e3f1730ace5d47b3d4ef80429ae2b053f
Size: 280264527 bytes
```

## 독립 예측 원칙

- 검증 연도보다 과거 데이터만 학습에 사용하는 forward OOF 방식을 사용했습니다.
- 평가 데이터의 정답이나 테스트 전체 분포를 사용하지 않습니다.
- 테스트 행 간 rolling·누적 집계를 생성하지 않습니다.
- 각 테스트 행과 학습 단계에서 저장한 artifact만으로 독립적으로 예측합니다.
- F 행의 예측은 JOA F-Regime075 anchor와 동일하게 유지합니다.

자세한 실행 방법은 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)를 참고하세요.
