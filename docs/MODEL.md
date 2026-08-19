# 최종 모델 구조

## 1. Hierarchical baseline

`asof_pitcher_n`, 커리어 성공률, 최근 1·3·5경기 성공률의 변동성을 이용해 커리어
성공률을 리그 평균 방향으로 동적 축소합니다. 직전 시즌 말 Train snapshot을 현재
누적값에서 빼 현재 시즌 표본과 성공률을 복원하고, 커리어 추정치와 시즌 추정치를
표본 신뢰도에 따라 결합합니다.

## 2. Residual 모델

정답 자체가 아니라 `control_success - hierarchical_base`를 학습합니다.

- v2, recency decay 0.55
- v3, recency decay 0.55
- v3, recency decay 0.30

각 채널은 여러 random seed의 CatBoost 평균을 사용합니다. v3에는 타자 시즌 기록,
투수 pitch mix, Trackman의 prior-season 요약과 상황 상호작용이 추가됩니다.

## 3. 실패유형과 adaptive gate

Train의 누적 as-of 값으로 middle·wild·reverse 보조 target을 생성해 세 분류기를
학습합니다. 선형 stack 이후 depth-3 gate가 모델 불일치, 표본수, 최근 변동성,
카운트·이닝·주자·LI를 보고 행별 residual 보정량을 조절합니다.

## 4. Futures regime

`game_type=F`는 퓨처스리그라는 별도 데이터 생성 과정이므로 공용 모델의 범주형
피처로만 처리하지 않고 F 모집단 전용 residual 전문가를 학습합니다.

- F v2: 전체 F 이력
- F v3 decay55: 최근 F 이력
- F v3 decay30: 전체·최근 F 전문가 혼합
- F middle·wild·reverse 전문가
- R/F 전환 residual gate

최종 선택본은 이 F 보정의 강도를 0.75로 적용합니다. Test에서는 현재 행의
`game_type`만 보며 다른 Test 행을 참조하지 않습니다.

## 5. Calibration

전역 calibration은 모든 행에 같은 고정 상수를 적용합니다. 개별 Test 행이나 Test
전체 통계를 읽지 않으며 최종 추론은 완전히 row-local입니다.
