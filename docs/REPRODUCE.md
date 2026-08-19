# 재현 방법

## 환경

- Python 3.11
- 6 CPU 이상 권장
- 학습 시 충분한 RAM 권장
- 최종 추론은 대회 제한인 10분 이내를 목표로 구성
- 최종 ZIP 다운로드를 위해 Git LFS 필요

```bash
git lfs install
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 데이터

공식 `train.csv`, `test.csv`, `trackman_history.csv`, `sample_submission.csv`를
저장소 루트의 `data/`에 둡니다. 데이터는 비공개 대회 자산이므로 Git에는 포함하지
않습니다.

## 정확한 최종 예측 재현

최종 ZIP에는 inference code와 모든 model asset이 들어 있습니다.

ZIP SHA-256:

```text
33b985a16d16bc868dd5e7778c3456c0740ae8a001fe2120cd46d6561c168796
```

```bash
unzip final/260818_F_regime075.zip -d run_final
cp -r data run_final/data
(cd run_final && python script.py)
```

출력: `run_final/output/submission.csv`

## 학습 코드

최종 단계는 다음 코드에 있습니다.

```text
final/training/train_f_regime_final.py
final/training/evaluate_crossfit_psych_profiles.py
final/training/evaluate_psych_residual_on_adaptive.py
final/training/evaluate_league_transition_gate.py
```

공용 전처리와 target 생성은 `src/`에 있습니다. 앞 단계의 seed ensemble, psych,
split-prior, stable channel 학습 코드는 모두 `archive/experiments/`에 보존했습니다.
주요 실행 순서는 다음과 같습니다.

```bash
PYTHONPATH=. python final/training/build_trackman_features.py
PYTHONPATH=. python final/training/train_hierarchical_stack.py
PYTHONPATH=. python final/training/train_seed_ensemble.py
PYTHONPATH=. python final/training/train_psych_latent_final.py
PYTHONPATH=. python final/training/train_psych_regime_final.py
PYTHONPATH=. python final/training/train_stable_minimax_final.py
PYTHONPATH=. python final/training/train_f_regime_final.py
```

CatBoost seed, iteration, decay 및 blend 상수는 각 스크립트와 최종 ZIP의
`model/manifest.json`, `model/f_regime_meta.json`에 고정되어 있습니다.

## 로컬 평가

후보 생성 코드가 `y22/p22/base22/pitcher22`부터 2024까지 포함한 NPZ를 만들면:

```bash
python evaluation/evaluate_local_v2.py candidate.npz --name candidate --verbose
```

후보 생성과 평가를 연속 실행하려면:

```bash
python evaluation/run_and_score.py experiment.py candidate.npz --name candidate --verbose
```

평가기의 `예측 점수`는 소수의 리더보드 anchor로 환산한 참고값이며, 실제 모델
선택은 세 연도 forward gain, pitcher-cluster bootstrap CI, pooled BSS를 우선합니다.

## 규정 검사

```bash
python evaluation/audit_submission_compliance.py final/260818_F_regime075.zip
python evaluation/check_row_independence.py final/260818_F_regime075.zip
```

행 하나, 부분집합, shuffle된 Test에서도 같은 행의 예측값이 동일해야 통과합니다.
