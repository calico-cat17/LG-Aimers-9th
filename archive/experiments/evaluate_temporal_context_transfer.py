"""Test whether previous-season context residuals transfer to the next season."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'data/train.csv',low_memory=False)

def score(y,p):
    return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))

def load(year):
    folder=ROOT/'old/experiments/result_dirs/v3_multisplit'
    f=folder/f'{year}.npz' if year<2024 else ROOT/'old/experiments/model_v3_2024.npz'
    z=np.load(f); frame=raw.loc[raw.season.eq(year)].reset_index(drop=True).copy()
    assert len(frame)==len(z['p'])
    return frame,z['y'].astype(float),z['p'].astype(float)

def contexts(d):
    out=pd.DataFrame(index=d.index)
    out['count']=d.balls_before.astype(str)+'-'+d.strikes_before.astype(str)
    out['base']=d.base_state.astype(str)
    out['count_base']=out['count']+'|'+out['base']
    out['inning_bin']=pd.cut(d.inning,[0,3,6,9,99],labels=False,include_lowest=True).fillna(-1).astype(int).astype(str)
    out['score_bin']=pd.cut(d.score_diff_pitcher_team,[-99,-4,-2,-1,1,2,4,99],labels=False,include_lowest=True).fillna(-1).astype(int).astype(str)
    out['li_bin']=pd.cut(d.li,[-1,.5,.8,1.2,2,4,999],labels=False,include_lowest=True).fillna(-1).astype(int).astype(str)
    out['runner_count']=d.num_runners_on.astype(str)
    out['hand']=d.pitcher_hand.astype(str)+'-'+d.batter_hand.astype(str)
    out['game_type']=d.game_type.astype(str)
    out['month']=d.game_month.astype(str)
    out['dayofweek']=d.game_dayofweek.astype(str)
    out['top_bottom']=d.top_bottom.astype(str)
    out['inning_score']=out.inning_bin+'|'+out.score_bin
    out['pressure']=out['li_bin']+'|'+out['runner_count']+'|'+out['count']
    out['team']=d.pitcher_team_id.astype(str)
    out['pitcher']=d.pitcher_id.astype(str)
    out['batter']=d.batter_id.astype(str)
    out['pitcher_batterhand']=out['pitcher']+'|'+d.batter_hand.astype(str)
    out['pitcher_count']=out['pitcher']+'|'+out['count']
    out['pitcher_base']=out['pitcher']+'|'+out['base']
    out['batter_pitcherhand']=out['batter']+'|'+d.pitcher_hand.astype(str)
    out['matchup']=out['pitcher']+'|'+out['batter']
    out['pitcher_batterteam']=out['pitcher']+'|'+d.batter_team_id.astype(str)
    out['pitcher_team_batterhand']=out['team']+'|'+d.batter_hand.astype(str)
    out['team_count']=out['team']+'|'+out['count']
    out['team_base']=out['team']+'|'+out['base']
    out['batter_count']=out['batter']+'|'+out['count']
    out['batter_base']=out['batter']+'|'+out['base']
    out['pitcher_game']=out['pitcher']+'|'+out['game_type']
    out['pitcher_score']=out['pitcher']+'|'+out['score_bin']
    out['pitcher_li']=out['pitcher']+'|'+out['li_bin']
    out['pitcher_inning']=out['pitcher']+'|'+out['inning_bin']
    out['month_hand']=out['month']+'|'+out['hand']
    out['month_count']=out['month']+'|'+out['count']
    out['month_pressure']=out['month']+'|'+out['li_bin']+'|'+out['runner_count']
    out['team_score']=out['team']+'|'+out['score_bin']
    return out

def table(train_ctx,resid,col,alpha):
    t=pd.DataFrame({'key':train_ctx[col], 'r':resid}).groupby('key').r.agg(['mean','count'])
    t['value']=t['mean']*t['count']/(t['count']+alpha)
    return t.value

data={yr:load(yr) for yr in (2022,2023,2024)}
groups=['count','base','count_base','inning_bin','score_bin','li_bin','runner_count','hand','game_type','month','dayofweek','top_bottom','inning_score','pressure','team','pitcher','batter','pitcher_batterhand','pitcher_count','pitcher_base','batter_pitcherhand','matchup','pitcher_batterteam','pitcher_team_batterhand','team_count','team_base','batter_count','batter_base','pitcher_game','pitcher_score','pitcher_li','pitcher_inning','month_hand','month_count','month_pressure','team_score']

for src,dst in ((2022,2023),(2023,2024)):
    a,ya,pa=data[src]; b,yb,pb=data[dst]
    ca,cb=contexts(a),contexts(b); resid=ya-pa
    print('\ntransfer',src,'->',dst,'base',score(yb,pb))
    best=(score(yb,pb),None)
    for alpha in (100,300,1000,3000,10000):
      corrections=[]
      for g in groups:
        corrections.append(cb[g].map(table(ca,resid,g,alpha)).fillna(0).to_numpy())
      C=np.column_stack(corrections)
      for scale in (.1,.2,.35,.5,.75,1.0):
        # Mean of weak context experts; each is shrunk independently.
        pred=pb+scale*C.mean(1)
        s=score(yb,pred)
        if s>best[0]: best=(s,(alpha,scale,'all'))
      # Each individual channel is also checked to expose transferable effects.
      for j,g in enumerate(groups):
        for scale in (.25,.5,1):
          s=score(yb,pb+scale*C[:,j])
          if s>best[0]: best=(s,(alpha,scale,g))
    print('best',best)

# Strict selection: configurations are selected on 2022->2023, then reported once on 2023->2024.
a,ya,pa=data[2022]; b,yb,pb=data[2023]; c,yc,pc=data[2024]
ca,cb,cc=contexts(a),contexts(b),contexts(c)
choices=[]
for alpha in (100,300,1000,3000,10000,30000):
 for g in groups:
  corr=cb[g].map(table(ca,ya-pa,g,alpha)).fillna(0).to_numpy()
  for scale in (.1,.2,.35,.5,.75,1): choices.append((score(yb,pb+scale*corr),alpha,g,scale))
print('\nselected on 2023',max(choices))
_,alpha,g,scale=max(choices)
corr=cc[g].map(table(cb,yb-pb,g,alpha)).fillna(0).to_numpy()
print('untouched 2024 base/new',score(yc,pc),score(yc,pc+scale*corr),'config',alpha,g,scale)

# Same hyperparameter must improve both transfers; rank by the worse improvement.
base23,base24=score(yb,pb),score(yc,pc); stable=[]
for alpha in (100,300,1000,3000,10000,30000):
 for g in groups:
  c23=cb[g].map(table(ca,ya-pa,g,alpha)).fillna(0).to_numpy()
  c24=cc[g].map(table(cb,yb-pb,g,alpha)).fillna(0).to_numpy()
  for scale in (.05,.1,.2,.35,.5,.75,1):
   d23=score(yb,pb+scale*c23)-base23; d24=score(yc,pc+scale*c24)-base24
   stable.append((min(d23,d24),(d23+d24)/2,d23,d24,alpha,g,scale))
print('stable top')
for row in sorted(stable,reverse=True)[:12]: print(row)
print('best stable by group')
for group in groups:
 rows=[r for r in stable if r[5]==group]
 if rows:print(group,max(rows))

# Apply the strictly previous-season hand correction to every current 2024 candidate.
hand24=cc['hand'].map(table(cb,yb-pb,'hand',1000)).fillna(0).to_numpy()
print('\n2024 candidate + transferred hand expert')
for f in [ROOT/'old/experiments/adaptive_gate_2024.npz',ROOT/'expanded_blend_2024.npz',ROOT/'old/experiments/group_meta_alpha100.npz']:
 z=np.load(f); key='p2024' if 'p2024' in z else 'p'; pred=z[key].astype(float)
 print(f.stem,score(yc,pred),score(yc,pred+.5*hand24))
