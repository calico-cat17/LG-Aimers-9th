"""Prior-only advanced transforms for rate, composition, missingness, and count state."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

COUNT_MAP={'0-0':'neutral','1-0':'batter_favored','2-0':'batter_favored','3-0':'batter_favored','0-1':'pitcher_favored','1-1':'neutral','2-1':'batter_favored','3-1':'batter_favored','0-2':'pitcher_favored','1-2':'pitcher_favored','2-2':'neutral','3-2':'full_count'}

def league_priors(history,target_year):
    last=history.loc[history.season.eq(target_year-1)]
    if len(last)==0:last=history
    cols=['control_success','asof_pitcher_middle_rate','asof_pitcher_reverse_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_batter_success_rate','asof_batter_middle_rate']
    return {c:float(pd.to_numeric(last[c],errors='coerce').mean()) for c in cols}

def _we_design(d):
    score=pd.to_numeric(d.score_diff_pitcher_team,errors='coerce').fillna(0).clip(-15,15);inn=pd.to_numeric(d.inning,errors='coerce').fillna(0).clip(0,15);outs=pd.to_numeric(d.outs_before,errors='coerce').fillna(0);run=pd.to_numeric(d.num_runners_on,errors='coerce').fillna(0)
    return np.c_[score,abs(score),inn,outs,run,score*inn,run*(3-outs),np.maximum(inn-6,0)*score]

def build_advanced_features(raw,history,season_n=None):
    r=raw.reset_index(drop=True);target=int(pd.to_numeric(r.season).mode().iloc[0]);pri=league_priors(history,target);x=pd.DataFrame(index=r.index);n=pd.to_numeric(r.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0);bn=pd.to_numeric(r.asof_batter_n,errors='coerce').fillna(0).clip(lower=0);mixn=pd.to_numeric(r.asof_pitcher_pitchmix_n,errors='coerce').fillna(0).clip(lower=0)
    # 1) Previous-season league environment centering.
    mapping={'asof_pitcher_success_rate':'control_success','asof_pitcher_middle_rate':'asof_pitcher_middle_rate','asof_pitcher_reverse_rate':'asof_pitcher_reverse_rate','asof_pitcher_ball_rate':'asof_pitcher_ball_rate','asof_pitcher_strike_rate':'asof_pitcher_strike_rate','asof_batter_success_rate':'asof_batter_success_rate','asof_batter_middle_rate':'asof_batter_middle_rate'}
    for c,p in mapping.items():x['league_centered_'+c.removeprefix('asof_')]=pd.to_numeric(r[c],errors='coerce')-pri[p]

    # 2) Pitch composition log-ratios, reliability-weighted variants included.
    eps=.01;fast=pd.to_numeric(r.asof_pitcher_fastball_rate,errors='coerce');brk=pd.to_numeric(r.asof_pitcher_breaking_rate,errors='coerce');off=pd.to_numeric(r.asof_pitcher_offspeed_rate,errors='coerce');rel=mixn/(mixn+100)
    x['log_fast_break']=np.log((fast+eps)/(brk+eps));x['log_fast_off']=np.log((fast+eps)/(off+eps));x['log_break_off']=np.log((brk+eps)/(off+eps))
    for c in ('log_fast_break','log_fast_off','log_break_off'):x[c+'_reliable']=x[c]*rel

    # 3) Smoothed failure composition and contrasts.
    prior=pri['control_success'];middle=pd.to_numeric(r.asof_pitcher_middle_rate,errors='coerce');reverse=pd.to_numeric(r.asof_pitcher_reverse_rate,errors='coerce');ball=pd.to_numeric(r.asof_pitcher_ball_rate,errors='coerce');strike=pd.to_numeric(r.asof_pitcher_strike_rate,errors='coerce');success=pd.to_numeric(r.asof_pitcher_success_rate,errors='coerce')
    failure=(1-success).clip(lower=.01);shrunk_failure=(failure*n+(1-prior)*100)/(n+100);x['middle_failure_share']=(middle*n+pri['asof_pitcher_middle_rate']*100)/(n+100)/shrunk_failure;x['reverse_failure_share']=(reverse*n+pri['asof_pitcher_reverse_rate']*100)/(n+100)/shrunk_failure;x['ball_middle_gap']=ball-middle;x['strike_success_gap']=strike-success;x['success_failure_margin']=success-middle-reverse

    # 4) Missingness is an explicit history state.
    recent=[f'asof_pitcher_prev{k}_game_success_rate' for k in (1,3,5)];x['recent_missing_count']=r[recent].isna().sum(1);x['recent_all_missing']=r[recent].isna().all(1).astype('int8');x['pitcher_history_missing']=(n==0).astype('int8');x['batter_history_missing']=(bn==0).astype('int8');x['pitchmix_missing']=(mixn==0).astype('int8');x['history_missing_count']=x[['pitcher_history_missing','batter_history_missing','pitchmix_missing']].sum(1)

    # 5) Current-season share supplied from leakage-safe snapshot delta.
    if season_n is None:season_n=np.zeros(len(r))
    sn=np.asarray(season_n,float);pre=np.maximum(n.to_numpy()-sn,0);x['season_log_n']=np.log1p(sn);x['preseason_log_n']=np.log1p(pre);x['season_history_share']=sn/(n.to_numpy()+1);x['season_reliability80']=sn/(sn+80);x['career_to_season_ratio']=np.log1p(pre)-np.log1p(sn)

    # 6) Win-expectancy residual relative to score/inning/outs/runners only.
    pitcher_home=r.top_bottom.astype(str).eq('T');pwe=np.where(pitcher_home,r.home_win_expectancy,r.away_win_expectancy)/100
    hh=history.copy();hh_home=hh.top_bottom.astype(str).eq('T');hy=np.where(hh_home,hh.home_win_expectancy,hh.away_win_expectancy)/100;we=Ridge(alpha=1000).fit(_we_design(hh),hy);expected=np.clip(we.predict(_we_design(r)),0,1);x['pitcher_win_expectancy']=pwe;x['state_expected_win']=expected;x['win_expectancy_residual']=pwe-expected;x['absolute_expectancy_surprise']=abs(pwe-expected)

    # 7) Pitcher-perspective count class and high-value interactions.
    count=r.balls_before.astype(int).astype(str)+'-'+r.strikes_before.astype(int).astype(str);state=count.map(COUNT_MAP).fillna('unknown');same=pd.to_numeric(r.pitcher_hand,errors='coerce').eq(pd.to_numeric(r.batter_hand,errors='coerce'))
    x['count_advantage']=state;x['count_numeric_advantage']=state.map({'pitcher_favored':1,'neutral':0,'batter_favored':-1,'full_count':-0.5}).fillna(0);x['count_platoon']=state+'|'+np.where(same,'same','opposite');x['count_fastball']=x.count_numeric_advantage*fast;x['count_breaking']=x.count_numeric_advantage*brk;x['count_offspeed']=x.count_numeric_advantage*off;x['fullcount_fastball']=state.eq('full_count')*fast;x['fullcount_breaking']=state.eq('full_count')*brk;x['fullcount_offspeed']=state.eq('full_count')*off
    return x.replace([np.inf,-np.inf],np.nan)

CAT_ADVANCED=['count_advantage','count_platoon']
