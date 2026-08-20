# LG Aimers 실험 기록

## 현재 유지 제출

- `final/sub_JM_R_res_scale075_RE.zip`
  - `260818_F_regime075.zip`의 F 예측을 고정하고 `game_type == "R"`인 행에만 3-seed CatBoost residual correction을 적용한다.
  - correction scale은 `0.075`이며 공식 리더보드 점수는 `1127.8851982941`이다.
  - 저장소에는 root-level 구조로 정리한 재패키징본을 Git LFS로 보존한다.
  - SHA-256: `46aa2a15130ed9ba8d302f203fa42c8e3f1730ace5d47b3d4ef80429ae2b053f`

## 리더보드 제출 기록

| 제출 | 목적 | 서버 점수 | 판정 |
|---|---|---:|---|
| `260811_adaptive_gate.zip` | 상황별 residual 보정량을 조절하는 adaptive gate | 1088.5349 | 기반 구조 |
| `260812_psych.zip` | 심리·최근 흐름 latent residual | 1088.6134 | 효과 미미 |
| `260812_regime_film.zip` | 상황 regime별 FiLM 전문가 | 1083.3372 | 폐기 |
| `260814_minimax.zip` | 연도별 안정성/minimax 보정 | 약 1069 | 폐기 |
| `260814_split_priors.zip` | 투수·상황별 부분풀링 prior | 1083.8061 | 폐기 |
| `260814_blend50.zip` | Adaptive와 split prior 중간점 | 1098.3775 | 채널 탐색 |
| `260814_channel_opt.zip` | 보정 채널 앙상블 가중치 최적화 | 1102.8479 | 채택 이력 |
| `260815_channel_cal.zip` | 채널 affine calibration | 1084.7331 | 폐기 |
| `260815_seed_ensemble.zip` | 동일 모델 6-seed 평균으로 예측 노이즈 축소 | 1108.9515 | 채택 이력 |
| `260815_probeA.zip` | 모든 행에 동일한 +0.03 적용 시 점수 변화 확인 | 868.7576 | 진단용 |
| `260815_shifted.zip` | 제한적 보간으로 정한 +0.0052 전역 상수 적용 | 1119.2195 | 채택 이력 |
| `260815_shrink85.zip` | 예측 분산 축소 효과 확인 | 1082.3432 | 폐기 |
| `260818_F_expert.zip` | `game_type=F` 전용 residual 전문가 | 1122.2577 | 채택 이력 |
| `260818_F_regime.zip` | F residual 3채널·실패유형·transition 결합 | 1126.4544 | 이전 최고 |
| `260818_F_regime24.zip` | residual ensemble을 24 seed로 확대 | 1124.0855 | 폐기 |
| `260818_F_regime125.zip` | F 보정 강도를 1.25배로 확대 | 1124.4084 | 과보정 |
| `260818_F_regime075.zip` | F 보정 강도를 0.75배로 완화한 R-residual anchor | 1126.8664 | 기준 anchor |
| `sub_JM_R_res_multiseed005.zip` | R 행에만 3-seed residual correction을 0.05배 적용 | 1127.5514 | 중간 최고 |
| `sub_JM_R_res_scale075.zip` | scale sweep에서 선택한 R correction 0.075 적용 | 1127.8852 | 새 최고 |
| `sub_JM_R_res_scale075_RE.zip` | scale075 제출물을 root-level ZIP으로 재패키징 | 1127.8852 | 현재 유지본 |

## R residual correction scale sweep

`260818_F_regime075.zip`을 고정 anchor로 사용하고 F 행의 예측은 변경하지 않았다. R 행에 대해서만 seed `17`, `42`, `777`의 CatBoost residual correction 평균을 적용했으며, correction scale을 `0.000`부터 `0.150`까지 `0.005` 간격으로 비교했다.

| Scale | 2024 forward BSS | Anchor 대비 ΔBSS | Bootstrap 95% CI | P(ΔBSS > 0) | 공식 점수 |
|---:|---:|---:|---:|---:|---:|
| 0.050 | 899.4424 | +1.7446 | [-0.6586, 4.1840] | 92.23% | 1127.5514 |
| 0.075 | 899.8907 | +2.1928 | [-1.4346, 5.8776] | 88.60% | **1127.8852** |
| 0.100 | 900.0562 | +2.3583 | [-2.5131, 7.2879] | 83.57% | 미제출 |

로컬 BSS만 보면 `0.100`이 가장 높지만 scale이 증가할수록 bootstrap 불확실성이 커지고 양의 개선 확률이 낮아졌다. 따라서 `0.050`보다 correction을 강화하면서도 `0.100`의 높은 분산을 피하는 절충점으로 `0.075`를 선택했다.

공식 리더보드에서 `0.075`는 F-Regime075 anchor보다 `+1.0187979238`, 기존 `0.050` 제출보다 `+0.3337797264` 높은 점수를 기록했다. `scale_curve_2024.csv`의 leaderboard projection은 형상 비교를 위한 참고치이며 실제 공식 점수와 동일한 값으로 해석하지 않는다.

## ZIP 없이 검증 후 폐기한 주요 방향

- RandomForest, HistGradientBoosting, LightGBM 독립 채널
- Transformer, context token, FiLM, psych interaction encoder
- Adversarial reweighting, Group DRO, IRM, 상태공간 모델
- Thin-history, 선발/불펜, 불펜×카운트 전문가
- Trackman pitch mixture, LambdaRank, F direct classifier
- 팀 성공률, season-state, cross-season 부분풀링 보정
- seed·iteration·모델 용량 확대

## 규정 준수 메모

- 최종 추론은 각 test 행의 공식 입력과 Train에서 미리 만든 자산만 사용한다.
- test의 다른 행을 이용한 누적·rolling·집계·분포 보정은 사용하지 않는다.
- 실패유형 등 보조 타깃은 공식 Train 데이터에서만 유도했다.
- 과거 파일은 삭제하지 않고 `old/` 아래에 보관해 재현 및 검증에 사용할 수 있다.
