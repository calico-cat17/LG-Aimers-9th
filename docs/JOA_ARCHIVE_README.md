# JOA 실험 아카이브

LG Aimers 9기 Phase 2에서 진행한 제구 성공 확률 예측 실험 중, 재사용 가치가 있는
방법론과 최종 결과만 정리한 아카이브입니다. 원본 대회 데이터, 모델 가중치, 제출 ZIP,
대용량 OOF 배열은 포함하지 않습니다.

## 최종 결과

- 초기 주요 제출: Public BSS `703.9432`
- 최종 최고 제출: Public BSS `1150.3783882339`
- 최종 방식: Tensor Empirical Bayes 기준 예측과 R residual 예측의 Quadratic Anti-Blend
- 이론 예상: `1150.3783651466`
- 예상과 실제의 차이: 약 `0.000023`

## 문서 안내

| 문서 | 내용 |
|---|---|
| `notion_project_report.md` | 프로젝트 전체 과정과 회고 |
| `progress_1088_to_1119.md` | 계층 모델, 채널 결합, seed 평균 및 calibration 개선 과정 |
| `experiments/JM_JY_OFFICIAL_QUADRATIC_ANTIBLEND.md` | 최종 Anti-Blend의 Brier 이차식과 공식 결과 |
| `rule_interpretation.md` | 평가 데이터 행 독립성 및 공식 데이터 사용 원칙 |
| `phase3_provenance.md` | 피처·후처리·앙상블 설정의 도출 근거 |
| `../evaluation/REBUILD_1200_AUDIT_260831.md` | 마지막 1200 재구축 검토와 기각 근거 |

## 코드 안내

`../jm_official_quadratic/`에는 공식 점수 세 점으로 Brier 점수 곡선을 복원하고,
최적 anti-blend 가중치를 계산하는 코드와 제출 wrapper가 포함되어 있습니다.

코드는 방법론 보존을 위한 것이며 기반 모델 가중치와 원본 데이터는 저장소에 포함하지
않으므로 그대로 실행되는 완전 재현 패키지는 아닙니다.

## 규칙 준수

- test의 각 행을 독립적으로 예측
- test 내부의 다른 행을 이용한 집계·rolling·lag 미사용
- test 전체 평균·분포·빈도·순위 기반 보정 미사용
- 공식 train 및 공식 Trackman에서 생성한 고정 자산만 활용
