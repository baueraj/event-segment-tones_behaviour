from init_est import *
from analy_funs_est import *
import init_est, analy_funs_est

# other vars
stup = {'c1_inds': [0, 2, 4],
        'c2_inds': [1, 3, 5]}
        #'col_names': ['RT', 'acc', 'subj', 'cartoon', 'clip',
        #              'dist', 'peak', 'afterPeak', 'b41stBound']}

kwargs_prepSubjDat = {'fl_detrend': True,
                      'RTthresh': [True, 2.5]}

# pilot 2
allDat_pre = get_participant_data(paths_plt2, cartoonNames, propTrialsThresh)
allDat = prep_subj_data(allDat_pre, cs_withDat_plt2, paths_plt2, **kwargs_prepSubjDat)
tonesDat = prep_tone_timestamps_v2(paths_plt2, importPaths)
#tonesDat_test = prep_tone_timestamps_v1(paths_plt2)


# prep for LMER ===============================================================
# cartoon 1
dfs_all = []
for i_c in stup['c1_inds']:
    for i_s in range(allDat['RT'][i_c].shape[1]):
        s_RT = allDat['RT'][i_c].iloc[:,i_s]
        s_acc = allDat['acc'][i_c].iloc[:,i_s]
        s_chngResp = allDat['chngResp'][i_c].iloc[:,i_s]
        s_aftrIncorr = allDat['aftrIncorr'][i_c].iloc[:,i_s]
        df_subjDat = pd.DataFrame({'RT': s_RT, 'acc': s_acc,
                                   'chngResp': s_chngResp, 'aftrIncorr': s_aftrIncorr})
        df_subj = pd.DataFrame({'subj': [s_RT.name] * len(df_subjDat)})
        df_cartoon = pd.DataFrame({'cartoon': [0] * len(df_subjDat)})
        cClip = stup['c1_inds'].index(i_c) + 1 + 0
        df_cClip = pd.DataFrame({'clip': [cClip] * len(df_subjDat)})
        df_tones = tonesDat[i_c]
        dfs = [df_subj, df_subjDat, df_cartoon, df_cClip, df_tones]
        dfs_all.append(pd.concat(dfs, axis=1))

df_forLMER_c1 = pd.concat(dfs_all)
df_forLMER_c1.loc[df_forLMER_c1['dist'].isnull(), :] = np.nan

# cartoon 2
dfs_all = []
for i_c in stup['c2_inds']:
    for i_s in range(allDat['RT'][i_c].shape[1]):
        s_RT = allDat['RT'][i_c].iloc[:,i_s]
        s_acc = allDat['acc'][i_c].iloc[:,i_s]
        s_chngResp = allDat['chngResp'][i_c].iloc[:,i_s]
        s_aftrIncorr = allDat['aftrIncorr'][i_c].iloc[:,i_s]
        df_subjDat = pd.DataFrame({'RT': s_RT, 'acc': s_acc,
                                   'chngResp': s_chngResp, 'aftrIncorr': s_aftrIncorr})
        df_subj = pd.DataFrame({'subj': [s_RT.name] * len(df_subjDat)})
        df_cartoon = pd.DataFrame({'cartoon': [1] * len(df_subjDat)})
        cClip = stup['c2_inds'].index(i_c) + 1 + 3
        df_cClip = pd.DataFrame({'clip': [cClip] * len(df_subjDat)})
        df_tones = tonesDat[i_c]
        dfs = [df_subj, df_subjDat, df_cartoon, df_cClip, df_tones]
        dfs_all.append(pd.concat(dfs, axis=1))

df_forLMER_c2 = pd.concat(dfs_all)
df_forLMER_c2.loc[df_forLMER_c2['dist'].isnull(), :] = np.nan



# save for analy_LMER_est.R ===================================================
df_forLMER_c1.to_csv('./forLMER_' + cartoonNames[0].split('_')[1] + '_prepped.csv', index=False)
df_forLMER_c2.to_csv('./forLMER_' + cartoonNames[1].split('_')[1] + '_prepped.csv', index=False)
