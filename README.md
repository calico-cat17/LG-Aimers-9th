# LG Aimers 9기 — JOA 실험 및 최종 재현 코드

투구 직전까지 확인 가능한 정보로 `control_success` 확률을 예측한 프로젝트입니다.
평가는 Brier Skill Score를 사용합니다.

이 브랜치는 다음 세 가지를 한 번에 공개합니다.

1. 지금까지 수행한 실험과 채택·폐기 결과
2. 현재 선택한 `F_regime075`의 학습 코드, 추론 코드, 제출 ZIP
3. 실제 사용한 2022·2023·2024 forward local 평가 체계

## 핵심 결과

| 모델 | Public BSS | 설명 |
|---|---:|---|
| Adaptive Gate | 1088.5349 | 계층 baseline 위 상황별 residual gate |
| Seed Ensemble + Shift | 1119.2195 | 6-seed 평균과 전역 calibration |
| F Expert | 1122.2577 | Futures 행 전용 residual 전문가 |
| F Regime | 1126.4544 | F residual 3채널·실패유형·transition 결합 |
| **F Regime 0.75** | **최종 선택본** | F 보정 강도를 0.75로 완화한 최종 제출 패키지 |

전체 기록은 [archive/EXPERIMENTS.md](archive/EXPERIMENTS.md)와
[archive/experiment_log.csv](archive/experiment_log.csv)에 있습니다.

## 저장소 구조

```text
├── final/
│   ├── 260818_F_regime075.zip   # 동일 예측을 내는 최종 제출물(Git LFS)
│   ├── inference/               # 압축을 풀지 않고도 볼 수 있는 추론 소스
│   └── training/                # 최종 F-regime 학습 단계
├── src/                         # 전처리, feature, gate, 보조 target 모듈
├── evaluation/                  # local 평가기, LB 환산기, forward OOF
├── archive/experiments/         # 지금까지 수행한 전체 실험·학습 코드
├── docs/
│   ├── MODEL.md                 # 최종 모델 구조
│   ├── REPRODUCE.md             # 설치부터 학습·평가·추론까지
│   └── data_description.md      # 공식 데이터 설명
└── requirements.txt
```

## 빠른 재현

Python 3.11 환경을 권장합니다.

```bash
git clone -b JOA https://github.com/calico-cat17/LG-Aimers-9th.git
cd LG-Aimers-9th
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

대회에서 받은 파일을 다음과 같이 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── trackman_history.csv
└── sample_submission.csv
```

정확한 최종 제출물의 추론:

```bash
unzip final/260818_F_regime075.zip -d run_final
cp -r data run_final/data
cd run_final
python script.py
```

결과는 `run_final/output/submission.csv`에 생성됩니다. 대회 원본 데이터는 저장소에
포함하지 않습니다.

로컬 평가:

```bash
python evaluation/evaluate_local_v2.py \
  evaluation/anchors/psych_latent.npz \
  --name psych_latent --verbose
```

후보 NPZ 형식과 평가 기준은 [evaluation/CANDIDATE_FORMAT.md](evaluation/CANDIDATE_FORMAT.md)를
참고하세요.

## 검증 원칙

- 2019~`t-1` 학습 → `t` 예측 방식으로 2022·2023·2024를 검증합니다.
- 세 연도 모두 개선되지 않는 후보는 채택하지 않습니다.
- Test의 다른 행, Test 전체 분포, Test 내부 rolling·누적 집계는 사용하지 않습니다.
- Train에서 만든 snapshot·통계·보조 target만 추론 자산으로 사용합니다.
- 최종 ZIP은 행 단독·부분집합·shuffle 입력에서 동일한 예측을 내는지 검사했습니다.

상세 재현 절차는 [docs/REPRODUCE.md](docs/REPRODUCE.md)에 있습니다.
