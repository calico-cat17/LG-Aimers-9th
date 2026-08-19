exec(open('evaluate_psych_gate.py').read().split("parts={yr:load_year")[0])
parts={yr:load_year(yr) for yr in (2022,2023,2024)}
base_cols=['p2','p55','p30','middle','wild','reverse','model_std','model_range','old','log_pitcher_n','log_batter_n','li','inning','balls','strikes','runners','form','volatility']
def score(y,p):q=y.mean();return 1e5*(1-np.mean((y-p)**2)/(q*(1-q)))
X=pd.concat([parts[2022][2],parts[2023][2]],ignore_index=True);y=np.r_[parts[2022][1],parts[2023][1]];old=np.r_[parts[2022][3],parts[2023][3]];years=np.r_[np.full(len(parts[2022][1]),2022),np.full(len(parts[2023][1]),2023)];_,yv,xv,ov=parts[2024]
for cols_name,cols in [('base',base_cols),('psych',list(X.columns))]:
 for decay in [.1,.2,.3,.4,.55,.7,1.0]:
  m=CatBoostRegressor(iterations=400,depth=3,learning_rate=.02,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=330000+int(decay*100),thread_count=6,allow_writing_files=False,verbose=False);w=np.power(decay,2023-years);m.fit(X[cols],y-old,sample_weight=w,eval_set=(xv[cols],yv-ov),early_stopping_rounds=80,use_best_model=True);c=m.predict(xv[cols]);best=max((score(yv,np.clip(ov+s*c,1e-6,1-1e-6)),s) for s in [.5,.75,1,1.25]);print(cols_name,decay,m.get_best_iteration()+1,best,flush=True)
