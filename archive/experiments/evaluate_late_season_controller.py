"""Select a late-season-only correction on 2023, audit once on 2024."""
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;z=np.load(R/'group_channels_crossseason.npz');names=[str(x) for x in z['names']];raw=pd.read_csv(R/'data/train.csv',low_memory=False);r23=raw.loc[raw.season.eq(2023)].reset_index(drop=True);r24=raw.loc[raw.season.eq(2024)].reset_index(drop=True);l23=r23.game_month.to_numpy()>=8;l24=r24.game_month.to_numpy()>=8
def sc(y,p):return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))
b23=sc(z['y23'][l23],z['p23'][l23]);b24=sc(z['y24'][l24],z['p24'][l24]);rows=[]
for j,n in enumerate(names):
 for w in (-.5,-.35,-.2,-.1,-.05,-.02,.02,.05,.1,.2,.35,.5):
  d23=sc(z['y23'][l23],z['p23'][l23]+w*z['x23'][l23,j])-b23;d24=sc(z['y24'][l24],z['p24'][l24]+w*z['x24'][l24,j])-b24;rows.append((d23,d24,min(d23,d24),n,w))
print('base late',b23,b24);print('selected solely by 2023')
for row in sorted(rows,key=lambda x:x[0],reverse=True)[:20]:print(row)
print('stable late')
for row in sorted(rows,key=lambda x:(x[2],x[0]+x[1]),reverse=True)[:30]:print(row)
