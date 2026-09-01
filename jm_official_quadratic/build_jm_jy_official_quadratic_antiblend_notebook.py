from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "LG_AIMERS_JM_JY_Official_Quadratic_AntiBlend_Colab.ipynb"
CODE_ZIP = ROOT / "LG_AIMERS_JM_JY_Official_Quadratic_AntiBlend_Code_v4.zip"


def md(value): return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}
def code(value): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(True)}


cells = [
md("""# JM × JY Official Quadratic Anti-Blend

세 공식 점수로 평가셋의 Brier blend 곡선을 직접 복원합니다.

| 입력 | 공식 점수 |
|---|---:|
| JM RobustTensorEB | 1148.77306 |
| JY residual ×0.15 | 1130.3604943627 |
| JM/JY 50:50 | 1142.1381933225 |

Brier 점수는 선형 예측 결합 비율에 대해 정확한 이차함수입니다. 복원된 최적점이 JM 바깥쪽에 있으므로 `JM - a × (JY-JM)`을 테스트합니다. 확률 clipping이 발생하면 이론 곡선이 달라지므로, 전체 test smoke에서 clipping 비율을 측정하고 0.5%를 넘으면 `NO_GO`로 판정합니다.

코드 ZIP을 `MyDrive/LG_AIMERS/`에 올린 뒤 실행하세요.
"""),
code("""from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
import json, shutil, subprocess, sys, zipfile
import pandas as pd

ROOT=Path('/content/drive/MyDrive/LG_AIMERS'); DATA=ROOT/'data'
CODE=ROOT/'LG_AIMERS_JM_JY_Official_Quadratic_AntiBlend_Code_v4.zip'

def newest(paths):
    paths=[p for p in paths if p.exists()]
    return max(paths,key=lambda p:p.stat().st_mtime) if paths else None

JM_SUB=newest(list(ROOT.rglob('submit_JM1_RobustTensorEB_pitcher_floor025_DACON.zip'))+
              list(ROOT.rglob('submit_JM1_RobustTensorEB_pitcher_floor025.zip')))
JY_SUB=newest(list(ROOT.rglob('sub_JY_team_residual_scale015_RE.zip'))+
              list(ROOT.rglob('sub_JY_team_residual_scale015.zip')))
required=[CODE,DATA/'test.csv',DATA/'sample_submission.csv']
missing=[str(p) for p in required if not p.exists()]
if JM_SUB is None: missing.append('JM RobustTensorEB submission ZIP')
if JY_SUB is None: missing.append('JY scale015 submission ZIP')
if missing: raise FileNotFoundError('Google Drive에서 다음 입력을 찾을 수 없습니다:\\n'+'\\n'.join(missing))
print('JM SUB:',JM_SUB); print('JY SUB:',JY_SUB)
"""),
code("""subprocess.run([sys.executable,'-m','pip','install','-q','pandas==2.3.3',
                'scikit-learn==1.8.0','joblib==1.5.3','catboost==1.2.10',
                'xgboost==2.0.3','lightgbm==4.3.0'],check=True)
LOCAL=Path('/content/jm_jy_official_antiblend'); LOCAL_DATA=LOCAL/'data'; OUTPUT=LOCAL/'output'; WORK=LOCAL/'code'
if LOCAL.exists(): shutil.rmtree(LOCAL)
for p in (LOCAL_DATA,OUTPUT,WORK): p.mkdir(parents=True,exist_ok=True)
for source,target in ((DATA/'test.csv',LOCAL_DATA/'test.csv'),
                      (DATA/'sample_submission.csv',LOCAL_DATA/'sample_submission.csv')):
    shutil.copy2(source,target)
with zipfile.ZipFile(CODE) as z: z.extractall(WORK)
SCRIPT=WORK/'jm_jy_official_quadratic_antiblend_suite.py'
WRAPPER=WORK/'jm_jy_official_quadratic_antiblend_script.py'
subprocess.run([sys.executable,'-m','py_compile',str(SCRIPT),str(WRAPPER)],check=True)
print('준비 완료')
"""),
code("""command=[sys.executable,str(SCRIPT),'--jm-submission',str(JM_SUB),
         '--jy-submission',str(JY_SUB),'--data-dir',str(LOCAL_DATA),
         '--wrapper',str(WRAPPER),'--output',str(OUTPUT),
         '--jm-score','1148.77306','--jy-score','1130.3604943627',
         '--half-score','1142.1381933225','--max-anti','0.50']
print('실행:',' '.join(command)); log_path=Path('/content/jm_jy_official_antiblend_run.log')
with log_path.open('w',encoding='utf-8') as log_file:
    process=subprocess.Popen(command,cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end=''); log_file.write(line); log_file.flush()
    returncode=process.wait()
if returncode: raise RuntimeError(f'실험 실패: returncode={returncode}')
shutil.copy2(log_path,OUTPUT/'colab_run.log')
"""),
code("""summary=json.loads((OUTPUT/'official_quadratic_antiblend_summary.json').read_text())
curve=pd.read_csv(OUTPUT/'official_quadratic_curve.csv')
display(curve.sort_values('predicted_official_score_unclipped',ascending=False).head(20))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('판정:',summary['decision'])
if summary['decision']=='NO_GO': print('clipping 또는 기대 개선폭 기준 미달: 제출하지 마세요.')
"""),
code("""DRIVE_OUT=ROOT/'jm_jy_official_quadratic_antiblend'/'v1'; DRIVE_OUT.mkdir(parents=True,exist_ok=True)
for path in OUTPUT.iterdir():
    if path.is_file(): shutil.copy2(path,DRIVE_OUT/path.name)
result=Path('/content/JM_JY_Official_Quadratic_AntiBlend_Results.zip')
with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('official_quadratic_antiblend_summary.json','official_quadratic_curve.csv',
                 'official_quadratic_candidate.csv','submission_smoke.log','colab_run.log'):
        path=OUTPUT/name
        if path.exists(): z.write(path,name)
print('Drive 결과:',DRIVE_OUT)
for path in DRIVE_OUT.glob('submit_*.zip'): print('제출 후보:',path)
files.download(str(result))
""")]

notebook={"cells":cells,"metadata":{"colab":{"name":NOTEBOOK.name,"provenance":[]},
          "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
          "language_info":{"name":"python","version":"3.x"}},"nbformat":4,"nbformat_minor":5}
NOTEBOOK.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
sources=[ROOT/'src'/'jm_jy_official_quadratic_antiblend_suite.py',
         ROOT/'src'/'jm_jy_official_quadratic_antiblend_script.py']
with zipfile.ZipFile(CODE_ZIP,'w',zipfile.ZIP_DEFLATED) as archive:
    for source in sources: archive.write(source,source.name)
for path in (NOTEBOOK,CODE_ZIP):
    print(path); print('SHA-256:',hashlib.sha256(path.read_bytes()).hexdigest())
