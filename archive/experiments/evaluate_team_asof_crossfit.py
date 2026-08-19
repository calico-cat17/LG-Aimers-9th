"""Strict train-only team-asof lookup channel requested in phase 25a."""
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
raw=pd.read_csv('data/train.csv',low_memory=False);strict=np.load('evaluation/season_state_crossfit.npz');Y=(2021,2022,2023,2024)
def skill(y,p):r=y.mean();return 1e5*(1-np.mean((y-np.clip(p,1e-6,1-1e-6))**2)/(r*(1-r)))
def stack21():
 z=np.load('artifacts/unified_oof/2021.npz');main=.27358084*z['p_v2']+.26512224*z['p_v3_55']+.46129691*z['p_v3_30'];return np.clip(.0300329767+np.c_[main,z['risks']]@np.array([.93505266,-.00520129,.01091677,-.02528331]),1e-6,1-1e-6)
base={2021:stack21(),2022:strict['base22'],2023:strict['base23'],2024:strict['base24']};ys={y:raw.loc[raw.season.eq(y),'control_success'].to_numpy(float) for y in Y};rows={y:raw.loc[raw.season.eq(y)].reset_index(drop=True) for y in Y}
def feat(year,alpha):
 h=raw.loc[raw.season.lt(year)];prior=h.control_success.mean();r=rows[year];out=[]
 for key in ('pitcher_team_id','batter_team_id'):
  a=h.groupby(key).control_success.agg(['sum','count']);rate=(a['sum']+alpha*prior)/(a['count']+alpha);out.append(r[key].map(rate).fillna(prior).to_numpy())
 pr,br=out;same=(r.pitcher_hand.to_numpy()==r.batter_hand.to_numpy()).astype(float);return np.c_[pr,br,pr-br,pr+br-2*prior,same,(pr-br)*same,np.log1p(r.asof_pitcher_n.to_numpy()),np.log1p(r.asof_batter_n.to_numpy())]
best=None
for shrink in (300,1000,3000,10000,30000):
 X={y:feat(y,shrink) for y in Y};corr={}
 for target in (2022,2023,2024):
  years=tuple(y for y in Y if y<target);xx=np.concatenate([X[y] for y in years]);ee=np.concatenate([ys[y]-base[y] for y in years]);w=np.concatenate([np.full(len(ys[y]),.55**((target-1)-y)) for y in years]);mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1;m=Ridge(alpha=3000,fit_intercept=False).fit((xx-mu)/sd,ee,sample_weight=w);corr[target]=m.predict((X[target]-mu)/sd)
 for scale in (.02,.05,.08,.1,.15,.2,.25,.3,.4):
  gains={y:skill(ys[y],base[y]+scale*corr[y])-skill(ys[y],base[y]) for y in (2022,2023,2024)};row=(min(gains.values()),sum(gains.values()),shrink,scale,gains);best=row if best is None or row[:2]>best[:2] else best
print('BEST',best)

# Audit the selected correction on top of the actual 2024 F-regime prediction,
# then fit the production 2025 coefficients from all strict OOF years.
shrink,scale=best[2],best[3];X={y:feat(y,shrink) for y in Y}
def fit_for(years,target_x,target_year):
 xx=np.concatenate([X[y] for y in years]);ee=np.concatenate([ys[y]-base[y] for y in years]);w=np.concatenate([np.full(len(ys[y]),.55**((target_year-1)-y)) for y in years]);mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1;m=Ridge(alpha=3000,fit_intercept=False).fit((xx-mu)/sd,ee,sample_weight=w);return m.predict((target_x-mu)/sd),mu,sd,m.coef_
c24,_,_,_=fit_for((2021,2022,2023),X[2024],2024);fr=np.load('evaluation/f_regime_2024.npz');print('OVER_F_REGIME',skill(fr['y'],fr['p']+scale*c24)-skill(fr['y'],fr['p']))
for league in ('R','F'):
 mask=rows[2024].game_type.eq(league).to_numpy();cc=np.where(mask,c24,0.0)
 print('OVER_F_REGIME',league,[(s,skill(fr['y'],fr['p']+s*cc)-skill(fr['y'],fr['p'])) for s in (-.20,-.15,-.10,-.08,-.05,-.03,-.02,-.01,.01,.02,.03,.05,.08)])
_,mu,sd,coef=fit_for(Y,X[2024],2025)
h=raw;prior=float(h.control_success.mean());tables={}
for key in ('pitcher_team_id','batter_team_id'):
 a=h.groupby(key).control_success.agg(['sum','count']);tables[key]=((a['sum']+shrink*prior)/(a['count']+shrink)).to_dict()
import pickle,json
with open('team_asof_tables.pkl','wb') as f:pickle.dump(tables,f)
np.savez('team_asof_meta.npz',mu=mu,sd=sd,coef=coef,scale=scale,prior=prior,shrink=shrink)
