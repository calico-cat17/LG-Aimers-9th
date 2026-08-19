"""Run a candidate generator and print the complete temporal local evaluation."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='후보 생성 코드 실행 후 TOTAL LOCAL SCORE 출력')
    ap.add_argument('script',help='수정한 실험/후보 생성 Python 파일')
    ap.add_argument('candidate',help='그 코드가 생성하는 NPZ 경로')
    ap.add_argument('--name',default=None);ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args();root=Path(__file__).resolve().parents[1];candidate=(root/args.candidate).resolve() if not Path(args.candidate).is_absolute() else Path(args.candidate)
    made=subprocess.run([sys.executable,args.script],cwd=root,text=True,capture_output=not args.verbose)
    if made.returncode:
        if made.stdout:print(made.stdout)
        if made.stderr:print(made.stderr,file=sys.stderr)
        raise SystemExit(made.returncode)
    if not candidate.exists():raise SystemExit(f'후보 NPZ가 생성되지 않았습니다: {candidate}')
    report=candidate.with_name(candidate.stem+'_report.json');cmd=[sys.executable,str(root/'evaluation/evaluate_local_v2.py'),str(candidate),'--name',args.name or candidate.stem,'--bootstrap',str(args.bootstrap),'--json-out',str(report)]
    if args.verbose:cmd.append('--verbose')
    subprocess.run(cmd,cwd=root,check=True)

if __name__=='__main__':main()
