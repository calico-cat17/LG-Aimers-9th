"""Conservative local evaluator calibrated with scarce 2025 leaderboard anchors.

Candidate NPZ (recommended):
  y23, p23, base23, pitcher23, y24, p24, base24, pitcher24
Here p is the candidate and base is the currently trusted model on the same rows.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parent

ANCHOR_ALIASES={
 'recent_residual':'260810_recent_residual.zip','pre_controller':'260810_pre_controller.zip',
 'momentum_blend':'260810_momentum_blend.zip','hierarchical_stack':'260811_hierarchical_stack.zip',
 'adaptive_gate':'260811_adaptive_gate.zip','cross_effects':'260811_cross_effects.zip',
 'cross_effects_2024_only':'260811_cross_effects.zip','psych_latent':'260812_psych.zip',
 'psych':'260812_psych.zip','psych_regime_film':'260812_regime_film.zip',
 'stable_minimax':'260814_minimax.zip','dynamic_stable_stack':'260814_minimax.zip',
 'channel_calibrated':'260815_channel_cal.zip','channel_calibration':'260815_channel_cal.zip',
 'f_expert':'260818_F_expert.zip','f_regime':'260818_F_regime.zip',
 'f_regime_local':'260818_F_regime.zip','f_regime24':'260818_F_regime24.zip',
}

def actual_site_score(name):
 key=str(name).lower().replace('.zip','')
 # Audit suffixes must not prevent known anchors from matching.
 for suffix in ('_audit','_v3','_2024_only'):
  if key.endswith(suffix):key=key[:-len(suffix)]
 submission=ANCHOR_ALIASES.get(key)
 if not submission:return None
 h=pd.read_csv(ROOT/'leaderboard_history.csv');v=h.loc[h.submission.eq(submission),'site_score']
 return None if v.empty else float(v.iloc[-1])

def bss(y,p):
 p=np.clip(np.asarray(p,float),1e-5,1-1e-5);y=np.asarray(y,float);r=y.mean()
 return 1e5*(1-np.mean((p-y)**2)/(r*(1-r)))

def cluster_ci(y,base,p,pitcher,n_boot=1000,seed=260812):
 delta=(np.asarray(base)-y)**2-(np.asarray(p)-y)**2;ids=np.asarray(pitcher);unique=np.unique(ids);rng=np.random.default_rng(seed)
 groups={q:delta[ids==q] for q in unique};boot=np.empty(n_boot)
 for i in range(n_boot):
  sample=rng.choice(unique,len(unique),replace=True);boot[i]=np.concatenate([groups[q] for q in sample]).mean()
 return float(delta.mean()),tuple(np.quantile(boot,[.025,.975]))

def site_model(local):
 h=pd.read_csv(ROOT/'leaderboard_history.csv').dropna(subset=['local_2024_score','site_score'])
 # Exclude random-OOF/meta baselines; only temporally evaluated model lineage is comparable.
 h=h.loc[h.family.isin(['controller_blend','hierarchical_stack','adaptive_gate','contextual_prior','psych_latent_residual','psych_regime_film','stable_minimax','channel_surface','channel_calibration','game_type_F_expert','game_type_F_regime'])].copy()
 x=h.local_2024_score.to_numpy(float);y=h.site_score.to_numpy(float)
 # Local BSS and leaderboard BSS use the same formula but different hidden years,
 # so expose a leaderboard-scale local score. Exact known anchors reproduce their
 # site scores; unseen candidates use smooth inverse-distance interpolation.
 def predict(q,xx=x,yy=y):
  exact=np.flatnonzero(np.isclose(xx,q,atol=1e-6))
  if len(exact):return float(np.mean(yy[exact]))
  take=np.argsort(np.abs(xx-q))[:min(4,len(xx))];dist=np.abs(xx[take]-q);w=1/(dist+20.0)**3
  return float(np.sum(w*yy[take])/np.sum(w))
 pred=predict(local)
 loo=[]
 for i in range(len(h)):
  loo.append(predict(x[i],np.delete(x,i),np.delete(y,i))-y[i])
 mae=float(np.mean(np.abs(loo)));q=float(np.quantile(np.abs(loo),.8))
 return pred,(pred-q,pred+q),mae,h[['submission','local_2024_score','site_score']]

def find_key(z,prefix,year):
 for k in (f'{prefix}{str(year)[-2:]}',f'{prefix}{year}'):
  if k in z:return k
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('--name',default=None);ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--json-out',default=None);ap.add_argument('--verbose',action='store_true',help='시즌별 BSS, CI, calibration anchors까지 표시');args=ap.parse_args()
 z=np.load(args.candidate,allow_pickle=True);name=args.name or Path(args.candidate).stem;rows=[]
 for year in (2022,2023,2024):
  ky,kp,kb,ki=[find_key(z,q,year) for q in ('y','p','base','pitcher')]
  if not ky or not kp:continue
  y,p=z[ky],z[kp];row={'year':year,'candidate_score':bss(y,p)}
  if kb:
   base=z[kb];row['base_score']=bss(y,base);row['gain']=row['candidate_score']-row['base_score'];row['mean_abs_change']=float(np.mean(np.abs(p-base)));row['mean_change']=float(np.mean(p-base))
   if ki:
    d,ci=cluster_ci(y,base,p,z[ki],args.bootstrap,260812+year);row['brier_gain']=d;row['cluster_ci_low'],row['cluster_ci_high']=ci;row['ci_pass']=ci[0]>0
  rows.append(row)
 if not rows:
  if 'y' not in z or 'p' not in z:raise SystemExit('NPZ must contain y/p or year-specific y23/p23 fields')
  rows=[{'year':2024,'candidate_score':bss(z['y'],z['p'])}]
 df=pd.DataFrame(rows);local=float(df.loc[df.year.eq(2024),'candidate_score'].iloc[0]);site,interval,loo_mae,anchors=site_model(local)
 # Exact score on all available forward rows. Unlike an average of yearly BSS,
 # this recomputes the Brier baseline after concatenating the seasons.
 pooled=None;pooled_base=None;pooled_gain=None
 yy=[];pp=[];bb=[]
 for year in (2022,2023,2024):
  ky,kp,kb=[find_key(z,q,year) for q in ('y','p','base')]
  if ky and kp:yy.append(z[ky]);pp.append(z[kp])
  if kb:bb.append(z[kb])
 if yy:
  pooled=bss(np.concatenate(yy),np.concatenate(pp))
  if len(bb)==len(yy):pooled_base=bss(np.concatenate(yy),np.concatenate(bb));pooled_gain=pooled-pooled_base
 gains=df['gain'].dropna() if 'gain' in df else pd.Series(dtype=float);ci_ok=bool(df.get('ci_pass',pd.Series([False])).all());worst=float(gains.min()) if len(gains) else None
 # Diversity is useful only when it also lowers loss: report how different the
 # candidate errors and corrections are from the trusted base.
 diversity=[]
 for year in (2022,2023,2024):
  ky,kp,kb=[find_key(z,q,year) for q in ('y','p','base')]
  if not (ky and kp and kb):continue
  y0,p0,b0=np.asarray(z[ky],float),np.asarray(z[kp],float),np.asarray(z[kb],float)
  err_corr=float(np.corrcoef(y0-p0,y0-b0)[0,1]) if len(y0)>1 else float('nan')
  diversity.append({'year':year,'error_correlation':err_corr,'prediction_delta_std':float(np.std(p0-b0))})
 # Upload gate requires temporal consistency and statistical evidence, not site regression alone.
 stability_gap=float(gains.max()-gains.min()) if len(gains)>=2 else None
 # Site history proves that a higher 2024 score alone is not transferable.
 # PASS therefore depends only on forward temporal evidence; site regression is diagnostic.
 temporal_gate=bool(len(gains)>=3 and worst>0 and ci_ok and pooled_gain is not None and pooled_gain>0 and stability_gap<=max(2.0,.5*abs(worst)))
 # Public history now contains several locally positive corrections that fell on
 # 2025. A candidate is upload-worthy only when its conservative calibrated
 # lower bound clears the best observed public score as well.
 best_public=float(pd.read_csv(ROOT/'leaderboard_history.csv').site_score.max())
 site_gate=bool(interval[0]>best_public)
 gate=bool(temporal_gate and site_gate)
 actual=actual_site_score(name)
 report={'name':name,'predicted_site_score':site,'actual_site_score':actual,'site_scale_interval':interval,'raw_total_bss':pooled,'raw_total_base_bss':pooled_base,'raw_total_gain':pooled_gain,'splits':df.to_dict('records'),'diversity':diversity,'worst_case_gain':worst,'temporal_gain_gap':stability_gap,'all_cluster_ci_positive':ci_ok,'temporal_gate':temporal_gate,'site_gate':site_gate,'best_public_score':best_public,'site_model_loo_mae':loo_mae,'upload_gate':'PASS' if gate else 'HOLD'}
 print('\n=== SCORE ===')
 print(f'예측 점수: {site:.4f}')
 print(f'실제 사이트 점수: {actual:.4f}' if actual is not None else '실제 사이트 점수: 미제출')
 print(f'SUBMIT = {report["upload_gate"]}')
 if args.verbose:
  print('\nSeason diagnostics:');print(df.to_string(index=False,float_format=lambda x:f'{x:.8f}'))
  print(f'\nexpected range={interval[0]:.2f}..{interval[1]:.2f}')
  print(f'RAW TOTAL BSS={pooled:.8f}  RAW BASE={pooled_base:.8f}  RAW GAIN={pooled_gain:+.8f}' if pooled is not None and pooled_base is not None else f'RAW TOTAL BSS={pooled}')
  print(f'worst_case_gain={worst}  all_cluster_ci_positive={ci_ok}  temporal_gain_gap={stability_gap}')
  if diversity:
   print('Diversity (lower error correlation means a more independent channel):')
   for d in diversity:print(f"  {d['year']}: error_corr={d['error_correlation']:.6f}  delta_std={d['prediction_delta_std']:.6f}")
  print(f'calibration LOO_MAE={loo_mae:.2f}')
  if not gate:print(f'HOLD 이유: temporal_gate={temporal_gate}, site_gate={site_gate}; 예상 하한이 현재 최고 {best_public:.4f}를 넘어야 제출')
  print('\nServer calibration anchors:');print(anchors.to_string(index=False))
 if args.json_out:Path(args.json_out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
