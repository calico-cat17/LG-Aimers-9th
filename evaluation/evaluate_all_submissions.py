"""Honest leave-one-submission-out audit of site-score calibration."""
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent
CURRENT_ANCHOR='260818_F_regime.zip'

# Submissions built but not yet scored. A blend entry is an exact convex mix of
# two scored submissions, so its local 2024 score is reconstructed from their
# saved forward predictions instead of being re-run.
PENDING=[]

def predict(q,x,y):
    take=np.argsort(np.abs(x-q))[:min(4,len(x))]
    d=np.abs(x[take]-q);w=1/(d+20.0)**3
    return float(np.sum(w*y[take])/np.sum(w))

def bss(y,p):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5);y=np.asarray(y,float);r=y.mean()
    return 1e5*(1-np.mean((p-y)**2)/(r*(1-r)))

def pending_rows():
    """Local 2024 score of each unscored submission, plus its exact identity value."""
    rows=[]
    for spec in PENDING:
        a,b,alpha=spec['blend']
        pa,pb=ROOT/a,ROOT/b
        if not pa.exists() or not pb.exists():continue
        za,zb=np.load(pa),np.load(pb)
        if 'p24' not in za.files or 'p24' not in zb.files:continue
        mix=alpha*za['p24'].astype(float)+(1-alpha)*zb['p24'].astype(float)
        y=za['y24'].astype(float)
        # S(aA+(1-a)B) = a*S_A + (1-a)*S_B + a(1-a)*G is exact on the site rows;
        # only G is estimated, here from the same forward-2024 predictions.
        g=1e5*float(((za['p24'].astype(float)-zb['p24'].astype(float))**2).mean())/float(y.mean()*(1-y.mean()))
        sa,sb=spec['endpoints']
        rows.append({'submission':spec['submission'],'local_2024_score':bss(y,mix),
                     'identity':alpha*sa+(1-alpha)*sb+alpha*(1-alpha)*g,'G_est':g})
    return rows

def main():
    h=pd.read_csv(ROOT/'leaderboard_history.csv').dropna(subset=['local_2024_score','site_score']).reset_index(drop=True)
    x=h.local_2024_score.to_numpy(float);y=h.site_score.to_numpy(float);rows=[]
    for i,r in h.iterrows():
        if r.submission == CURRENT_ANCHOR:
            # This scored model defines the current conversion origin. It is not
            # honest to LOO-predict it from older, structurally different models.
            pred=float(y[i]); method='CURRENT ANCHOR'
        else:
            pred=predict(x[i],np.delete(x,i),np.delete(y,i)); method='LOO'
        rows.append((r.submission,pred,y[i],y[i]-pred,method))
    out=pd.DataFrame(rows,columns=['submission','predicted_site_score','actual_site_score','error','method'])
    extra=pending_rows()
    for e in extra:
        rows.append((e['submission'],predict(e['local_2024_score'],x,y),np.nan,np.nan,'PENDING'))
    shown=pd.DataFrame(rows,columns=['submission','predicted_site_score','actual_site_score','error','method'])
    print('\n=== 전체 제출 점수 비교 (LOO) ===')
    name_width=max(34,max(len(str(v)) for v in shown.submission))
    print(f'{"SUBMISSION":<{name_width}}  {"PREDICTED":>10}  {"ACTUAL":>10}  {"ERROR":>10}  {"METHOD":>14}')
    print(f'{"-"*name_width}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*14}')
    for r in shown.itertuples(index=False):
        if r.actual_site_score!=r.actual_site_score:
            print(f'{r.submission:<{name_width}}  {r.predicted_site_score:>10.4f}  {"미제출":>10}  {"-":>10}  {r.method:>14}')
        else:
            print(f'{r.submission:<{name_width}}  {r.predicted_site_score:>10.4f}  {r.actual_site_score:>10.4f}  {r.error:>+10.4f}  {r.method:>14}')
    loo=out[out.method.eq('LOO')]
    print(f'\n과거 LOO 평균 절대 오차(MAE): {loo.error.abs().mean():.4f}')
    print(f'현재 환산 기준: {CURRENT_ANCHOR} = 1126.4544')
    for e in extra:
        print(f'{e["submission"]} 항등식 예측(보간기와 무관, G만 추정): {e["identity"]:.4f}  (local 2024 = {e["local_2024_score"]:.4f}, G_est = {e["G_est"]:.1f})')
    out.to_csv(ROOT/'all_submission_calibration.csv',index=False)

if __name__=='__main__':main()
