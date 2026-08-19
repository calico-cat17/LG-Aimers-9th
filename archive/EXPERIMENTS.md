# LG Aimers 실험 기록

## 현재 유지 제출

- `submissions/260818_F_regime075.zip`
  - 기존 F-regime의 퓨처스리그 전용 residual·실패유형·transition 보정 강도를 0.75배로 완화한 제출.
  - 최상위 작업 폴더에는 이 ZIP과 대응 추론 패키지 `submission_f_regime075/`만 유지한다.

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
| `260818_F_regime075.zip` | F 보정 강도를 0.75배로 완화 | 사용자 지정 현재 유지본 | 유지 |

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
