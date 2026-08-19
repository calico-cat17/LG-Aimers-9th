"""Forward tests for physical-repeatability and state-change residual channels."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import attach_profile, load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive, standardize


def trackman_channel(rows, table):
    keys=pd.DataFrame({'pitcher_id':rows.pitcher_id.astype(str),'season':rows.season.astype(int)})
    z=keys.merge(table,on=['pitcher_id','season'],how='left',sort=False)
    out=pd.DataFrame(index=rows.index)
    n=pd.to_numeric(z.tm_n,errors='coerce').fillna(0)
    last_n=pd.to_numeric(z.tm_last_n,errors='coerce').fillna(0)
    out['tm_log_n']=np.log1p(n);out['tm_log_last_n']=np.log1p(last_n)
    out['tm_reliability']=n/(n+500);out['tm_last_reliability']=last_n/(last_n+200)
    out['tm_mapping_similarity']=pd.to_numeric(z.tm_mapping_similarity,errors='coerce').fillna(0)
    for c in ('extension','rel_height','rel_side','rel_speed','zone_speed','induced_vert_break','horz_break'):
        mean=pd.to_numeric(z[f'tm_{c}_mean'],errors='coerce')
        sd=pd.to_numeric(z[f'tm_{c}_std'],errors='coerce')
        last_mean=pd.to_numeric(z[f'tm_last_{c}_mean'],errors='coerce')
        last_sd=pd.to_numeric(z[f'tm_last_{c}_std'],errors='coerce')
        out[f'tm_{c}_cv']=sd/(mean.abs()+1e-3)
        out[f'tm_last_{c}_cv']=last_sd/(last_mean.abs()+1e-3)
        out[f'tm_{c}_center_shift']=(last_mean-mean)/(sd.abs()+1e-3)
        out[f'tm_{c}_stability_change']=(last_sd-sd)/(sd.abs()+1e-3)
    out['tm_release_area']=np.sqrt(
        pd.to_numeric(z.tm_rel_height_std,errors='coerce').pow(2)+
        pd.to_numeric(z.tm_rel_side_std,errors='coerce').pow(2))
    out['tm_last_release_area']=np.sqrt(
        pd.to_numeric(z.tm_last_rel_height_std,errors='coerce').pow(2)+
        pd.to_numeric(z.tm_last_rel_side_std,errors='coerce').pow(2))
    return out.replace([np.inf,-np.inf],np.nan)


def state_channel(rows):
    rec=rows[[f'asof_pitcher_prev{k}_game_success_rate' for k in (1,3,5)]].apply(pd.to_numeric,errors='coerce')
    mid=rows[[f'asof_pitcher_prev{k}_game_middle_rate' for k in (1,3,5)]].apply(pd.to_numeric,errors='coerce')
    career=pd.to_numeric(rows.asof_pitcher_success_rate,errors='coerce')
    n=pd.to_numeric(rows.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0)
    out=pd.DataFrame(index=rows.index)
    out['state_log_n']=np.log1p(n)
    out['state_level1']=rec.iloc[:,0]-career;out['state_level3']=rec.iloc[:,1]-career;out['state_level5']=rec.iloc[:,2]-career
    out['state_velocity']=rec.iloc[:,0]-rec.iloc[:,1]
    out['state_slow_velocity']=rec.iloc[:,1]-rec.iloc[:,2]
    out['state_acceleration']=rec.iloc[:,0]-2*rec.iloc[:,1]+rec.iloc[:,2]
    out['state_abs_shock']=out.state_velocity.abs()
    out['state_direction_agreement']=np.sign(out.state_velocity)*np.sign(out.state_slow_velocity)
    out['state_volatility']=rec.std(axis=1)
    out['middle_velocity']=mid.iloc[:,0]-mid.iloc[:,1]
    out['middle_acceleration']=mid.iloc[:,0]-2*mid.iloc[:,1]+mid.iloc[:,2]
    out['success_middle_shock']=out.state_velocity-out.middle_velocity
    # Reliability rises smoothly; no dataset-level test statistics are used.
    rel=n/(n+300)
    for c in list(out.columns):
        if c!='state_log_n':out[c+'_reliable']=out[c]*rel
    return out.replace([np.inf,-np.inf],np.nan)

def latent_pitch_channel(rows, table):
    keys=pd.DataFrame({'pitcher_id':rows.pitcher_id.astype(str),'season':rows.season.astype(int),
        'balls_before':rows.balls_before.astype(int),'strikes_before':rows.strikes_before.astype(int),
        'batter_hand':rows.batter_hand.map({1:'Left',2:'Right'}).fillna(rows.batter_hand.astype(str))})
    z=keys.merge(table,on=['pitcher_id','season','balls_before','strikes_before','batter_hand'],how='left',sort=False)
    out=z.drop(columns=['pitcher_id','season','balls_before','strikes_before','batter_hand'])
    n=pd.to_numeric(out.latent_pitch_n,errors='coerce').fillna(0)
    out['latent_log_n']=np.log1p(n);out['latent_reliability']=n/(n+100)
    probs=out[[f'latent_{g}_prob' for g in ('fastball','breaking','offspeed','other')]].clip(1e-6,1)
    out['latent_entropy']=-(probs*np.log(probs)).sum(1)
    out['latent_fast_vs_break']=probs.latent_fastball_prob-probs.latent_breaking_prob
    return out.replace([np.inf,-np.inf],np.nan)


def main():
    raw=pd.read_csv('data/train.csv',low_memory=False)
    tm=pd.read_csv('artifacts/trackman_prior_features.csv',dtype={'pitcher_id':str})
    latent=pd.read_csv('artifacts/latent_pitch_context.csv',dtype={'pitcher_id':str,'batter_hand':str})
    loaded={y:load_predictions(raw,y) for y in (2022,2023,2024)}
    r22,y22,x22,b22=loaded[2022];r23,y23,x23,b23=loaded[2023];r24,y24,x24,b24=loaded[2024]
    g23=fit_adaptive(x22,y22,b22,620023);a22=b22.copy();a23=np.clip(b23+g23.predict(x23),1e-6,1-1e-6)
    a24=np.load('old/experiments/adaptive_gate_2024.npz')['p'].astype(float)
    bases={2022:a22,2023:a23,2024:a24};ys={2022:y22,2023:y23,2024:y24};rows={2022:r22,2023:r23,2024:r24}
    channels={
      'trackman':{y:trackman_channel(rows[y],tm) for y in rows},
      'state':{y:state_channel(rows[y]) for y in rows},
      'latent':{y:latent_pitch_channel(rows[y],latent) for y in rows},
      'psych':{y:attach_profile(rows[y],raw.loc[raw.season.lt(y)],500.) for y in rows},
    }
    channels['combined']={y:pd.concat([channels['trackman'][y],channels['state'][y]],axis=1) for y in rows}
    channels['latent_state']={y:pd.concat([channels['latent'][y],channels['state'][y]],axis=1) for y in rows}
    channels['psych_latent']={y:pd.concat([channels['psych'][y],channels['latent'][y]],axis=1) for y in rows}
    channels['psych_latent_state']={y:pd.concat([channels['psych'][y],channels['latent'][y],channels['state'][y]],axis=1) for y in rows}
    channel_filter=os.environ.get('CHANNEL_FILTER')
    if channel_filter:channels={k:v for k,v in channels.items() if k in channel_filter.split(',')}
    scales=(.02,.05,.08,.10,.15,.20,.25,.30,.40,.50)
    for name,data in channels.items():
      for ridge in (100,300,1000,3000,10000,30000,100000):
        z22,z23=standardize(data[2022],data[2023]);m23=Ridge(alpha=ridge,fit_intercept=False).fit(z22,y22-a22);c23=m23.predict(z23)
        tr=pd.concat([data[2022],data[2023]],ignore_index=True);ztr,z24=standardize(tr,data[2024]);target=np.r_[y22-a22,y23-a23];w=np.r_[np.full(len(y22),.55),np.ones(len(y23))]
        m24=Ridge(alpha=ridge,fit_intercept=False).fit(ztr,target,sample_weight=w);c24=m24.predict(z24)
        s23=np.array([skill(y23,np.clip(a23+s*c23,1e-6,1-1e-6)) for s in scales]);s24=np.array([skill(y24,np.clip(a24+s*c24,1e-6,1-1e-6)) for s in scales])
        gain=np.minimum(s23-skill(y23,a23),s24-skill(y24,a24));j=int(gain.argmax())
        if gain[j]>0: print(name,ridge,scales[j],round(s23[j],4),round(s24[j],4),'gains',round(s23[j]-skill(y23,a23),4),round(s24[j]-skill(y24,a24),4),flush=True)
        if name=='psych_latent' and ridge==10000:
            np.savez_compressed('psych_latent_residual_2024.npz',y=y24,base=a24,
                p=np.clip(a24+.4*c24,1e-6,1-1e-6),correction=c24,scale=.4,ridge=ridge)

if __name__=='__main__':main()
