# 로컬 평가 v3 사용법

실험 코드가 다음 배열을 NPZ로 저장하게 합니다.

```python
np.savez_compressed(
    "evaluation/my_candidate.npz",
    y23=y23, p23=candidate23, base23=trusted23, pitcher23=pitcher_id23,
    y24=y24, p24=candidate24, base24=trusted24, pitcher24=pitcher_id24,
)
```

- `p23`, `p24`: 수정한 후보 모델 예측
- `base23`, `base24`: 같은 행에서 비교할 기존 기준 모델 예측
- `pitcher23`, `pitcher24`: cluster bootstrap에 사용할 투수 ID
- 반드시 시간 순서대로 학습한 OOF/forward 예측이어야 합니다.

실행:

```bash
./trainenv311/bin/python evaluation/evaluate_local_v2.py \
  evaluation/my_candidate.npz --name my_candidate \
  --json-out evaluation/my_candidate_report.json
```

`UPLOAD GATE: PASS`일 때만 제출 ZIP 생성을 검토합니다. 서버 예상치는 실제
라벨의 대체물이 아니며, 현재 소수 제출 이력에 대한 경험적 참고 범위입니다.

## 코드 수정 후 한 번에 실행

수정한 실험 코드가 위 형식의 NPZ를 저장하도록 한 뒤 다음 한 명령만 실행합니다.

```bash
./trainenv311/bin/python evaluation/run_and_score.py \
  evaluate_psych_film_expert.py evaluation/psych_regime_film.npz \
  --name psych_regime_film
```

이 명령은 후보 생성 코드를 먼저 실행하고 이어서 다음을 출력합니다.

- 시즌별 Brier Skill Score와 기준 대비 gain
- `SITE-SCALE LOCAL SCORE`: 기존 제출 앵커로 보정한 서버와 같은 1000점대 척도
- `RAW TOTAL BSS`: 2023·2024 행을 합쳐 다시 계산한 원래 BSS
- `RAW GAIN`: 같은 합산 데이터에서 기준 모델 대비 상승량
- `worst_case_gain`, 연도별 효과 차이, 투수 cluster bootstrap CI
- 저신뢰 서버 참고 추정치와 `PASS/HOLD`

대표 로컬 점수는 `SITE-SCALE LOCAL SCORE`입니다. 모델 선택 시에는 이 점수와 함께 `RAW GAIN`, `worst_case_gain`,
두 시즌의 `cluster_ci_low > 0` 여부를 우선합니다.
