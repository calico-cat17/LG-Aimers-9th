# Local evaluation

`evaluate_local_v2.py`는 2022·2023·2024 forward prediction을 한 번에 평가합니다.

필수 후보 형식:

```text
y22, p22, base22, pitcher22
y23, p23, base23, pitcher23
y24, p24, base24, pitcher24
```

판정은 다음을 함께 봅니다.

- 연도별 Brier Skill Score와 incumbent 대비 gain
- 세 연도 중 최악 gain
- 전체 forward 행을 합친 pooled BSS
- pitcher-cluster bootstrap 95% CI
- 기존 모델과의 error correlation
- 과거 실제 제출 점수를 이용한 참고용 site-scale 환산

`unified_oof/`에는 동일한 forward protocol의 residual 채널 예측이 들어 있습니다.
`leaderboard_history.csv`는 환산 anchor와 실험 기록이며 Test 입력으로 사용되지 않습니다.
