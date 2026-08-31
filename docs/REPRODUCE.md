# 재현 방법

## 환경

- Python 3.11
- 대회 평가 환경: Ubuntu 22.04, 6 vCPU, 28GB RAM, NVIDIA L4
- 최종 제출 ZIP 다운로드를 위해 Git LFS 필요

```bash
git lfs install
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 데이터와 외부 artifact

공식 데이터는 저장소 루트의 `data/`에 둡니다.

```text
data/train.csv
data/test.csv
data/trackman_history.csv
data/sample_submission.csv
```

최종 학습을 다시 수행하려면 다음 비공개/중간 artifact도 필요합니다.

```text
260818_F_regime075.zip
JOA_strict_F_regime_OOF_validated_full_gpu.zip
JOA_Candidate4_Strict_Blend_Results.zip
LG_AIMERS_JOA_R_Residual_Final_Trainer_Code_v1.zip
```

대회 데이터와 OOF·중간 모델 artifact는 Git에 포함하지 않습니다.

## 최종 제출 ZIP 검증

현재 JM 브랜치의 최종 제출물은 다음과 같습니다.

```text
final/sub_JM_R_res_scale075_RE.zip
```

```bash
shasum -a 256 final/sub_JM_R_res_scale075_RE.zip
unzip -tq final/sub_JM_R_res_scale075_RE.zip
```

예상 SHA-256과 파일 크기:

```text
46aa2a15130ed9ba8d302f203fa42c8e3f1730ace5d47b3d4ef80429ae2b053f
280264527 bytes
```

Git에서 내려받은 파일이 약 134바이트이고 내용이 `version https://git-lfs.github.com/spec/v1`로 시작한다면 LFS pointer만 받은 상태입니다. 다음 명령으로 실제 ZIP을 내려받습니다.

```bash
git lfs pull
```

## 로컬 추론 확인

```bash
JM_CHECK_DIR="$(mktemp -d /tmp/lg_aimers_jm_scale075.XXXXXX)"
unzip final/sub_JM_R_res_scale075_RE.zip -d "$JM_CHECK_DIR"
cp -R data "$JM_CHECK_DIR/data"
(cd "$JM_CHECK_DIR" && python script.py)
```

예상 출력:

```text
$JM_CHECK_DIR/output/submission.csv
```

실제 평가 서버에서는 `data/`와 `output/`이 자동으로 추가되므로 제출 ZIP 내부에는 다음 최상위 구조가 있어야 합니다.

```text
model/
script.py
requirements.txt
```

## Google Colab 재현

### 1. R residual 모델 학습과 0.05 패키지 생성

다음 노트북은 실제 최종 학습·패키징에 사용한 Colab 실행본입니다.

```text
notebooks/LG_AIMERS_JOA_R_Residual_Final_Trainer_Colab.ipynb
```

Google Drive의 `MyDrive/LG_AIMERS/` 아래에 데이터와 외부 artifact를 배치한 뒤 위에서 아래로 실행합니다. seed `17`, `42`, `777`의 residual 모델과 scale 0.05 기준 패키지가 생성됩니다.

### 2. Correction scale 0.075 선택과 패키지 생성

```text
notebooks/LG_AIMERS_JOA_R_Scale_Sweep_Colab.ipynb
```

1단계에서 생성한 모델/checkpoint를 사용해 2024 forward OOF에서 scale을 비교하고 0.075 패키지를 생성합니다. 검증 결과는 다음 파일에 기록돼 있습니다.

```text
reports/r_scale_sweep/scale_curve_2024.csv
reports/r_scale_sweep/scale_candidate_summary.csv
reports/r_scale_sweep/submission_manifest.csv
```

## 로컬 평가

후보 NPZ가 `y22/p22/base22/pitcher22`부터 2024까지 포함하는 경우:

```bash
python evaluation/evaluate_local_v2.py candidate.npz --name candidate --verbose
```

모델 선택은 단일 환산 점수보다 다음 항목을 우선합니다.

- 연도별 forward BSS
- anchor 대비 ΔBSS
- pitcher-cluster bootstrap CI
- `P(ΔBSS > 0)`
- F/R 그룹별 성능

## 규정 검사

```bash
python evaluation/audit_submission_compliance.py final/sub_JM_R_res_scale075_RE.zip
python evaluation/check_row_independence.py final/sub_JM_R_res_scale075_RE.zip
```

동일한 테스트 행은 단일 행, 부분집합, 전체 데이터, shuffle된 데이터에서 같은 예측을 반환해야 합니다. 테스트 전체 분포나 다른 테스트 행의 정보를 이용한 보정은 허용하지 않습니다.
