"""Small-data leaderboard emulator with explicit extrapolation uncertainty.

The five site submissions are too few for a trustworthy universal regressor.
Known submissions are audited exactly; new controller-family candidates are
estimated by both linear and saturating quadratic calibrations fitted only to
the three comparable public submissions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent

def fit_calibrators(history):
    h=history.query("family == 'controller_blend'").dropna(subset=['local_2024_score'])
    x=h.local_2024_score.to_numpy(float); y=h.site_score.to_numpy(float)
    return x,y,np.polyfit(x,y,1),np.polyfit(x,y,2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',nargs='?',default=str(ROOT/'candidates/season_adaptive.json'));args=ap.parse_args()
    hist=pd.read_csv(ROOT/'leaderboard_history.csv')
    print('\nKnown site results (audit table)')
    print(hist[['submission','site_score','seconds']].to_string(index=False))
    c=json.load(open(args.candidate,encoding='utf-8')); local=float(c['local_2024_score'])
    x,y,linear,quadratic=fit_calibrators(hist)
    pred_linear=float(np.polyval(linear,local));pred_quad=float(np.polyval(quadratic,local))
    lo=min(pred_linear,pred_quad);hi=max(pred_linear,pred_quad)
    forward=np.array(list(c['forward_scores'].values()),float);base=np.array(list(c['benchmark_forward_scores'].values()),float);delta=forward-base
    outside=max(0.,local-x.max()); span=x.max()-x.min(); extrapolation=outside/max(span,1e-9)
    # Do not collapse model disagreement to a falsely precise point estimate.
    print('\nCandidate:',c['name'])
    print(f"local_2024={local:.3f}")
    print('forward_scores=',c['forward_scores'])
    print('forward_delta=',dict(zip(c['forward_scores'],np.round(delta,1))))
    print(f"site_estimate_range={lo:.1f}..{hi:.1f}")
    print(f"extrapolation_ratio={extrapolation:.2f} (0=in calibrated range)")
    stable=bool(np.all(delta>0)); meaningful=lo>hist.site_score.max()+10
    print(f"all_forward_splits_improved={stable}")
    print(f"submission_gate={'PASS' if stable and meaningful else 'HOLD'}")
    if not meaningful: print('reason: conservative estimate is not 10 points above the current public best')

if __name__=='__main__':main()
