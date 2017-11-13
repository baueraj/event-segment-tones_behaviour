from init_est import *
from analy_funs_est import *
import init_est, analy_funs_est
for imPath_i in importPaths:
    sys.path.append(imPath_i)
import init_es
#plt.close('all')

# other vars
stup = {'c_inds': [1, 2]}

# pilot 2 data
allDat_plt2_pre = get_participant_data(paths_plt2, cartoonNames, propTrialsThresh)
allDat_plt2 = prep_subj_data(allDat_plt2_pre, cs_withDat_plt2, paths_plt2)



# plotting ====================================================================
# plot tone data on top of event boundaries (just 1 clip)

# setup vars: non-0-index labeling
stup_pl = {'c': 1,
           'c_clip': 1, # relative to cartoon
           'pGrp': 1,
           'smooth_win': 2,
           'smooth_stdev': 1}

dfDesign = pd.read_csv(paths_plt2[1])
dfDesign_peaks = pd.read_csv(paths_plt2[2])
i_c = (stup_pl['c']-1) * 3 + (stup_pl['c_clip']-1)
inds_c = [i_c*2, i_c*2+1]
dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                          .dropna(axis=0, how='any')
dfDes_peaks_c = dfDesign_peaks.iloc[:, stup_pl['c']-1].copy() \
                                   .dropna(axis=0, how='any')       

des_id = '{}_{}_onset'.format(cartoonNames[stup_pl['c']-1], stup_pl['c_clip'])
tonesTiming = dfDesign_c[des_id].copy()/1000
tonesTiming.name = 'onset'

evntBound = init_es.plot_mov_gaussian_events(init_es.allDat['aEventBounds'][stup_pl['c']].sum(axis=1),
                                        stup_pl['smooth_win'], stup_pl['smooth_stdev'], 'b', 1, 0, 1) \
                                        .fillna(value=0)

meanRT_pre = allDat_plt2['RT'][i_c].mean(axis=1)
meanRT_pre.name = 'RT'
meanRT = pd.concat([meanRT_pre, tonesTiming], axis=1)

meanAcc_pre = allDat_plt2['acc'][i_c].mean(axis=1)
meanAcc_pre.name = 'acc'
meanAcc = pd.concat([meanAcc_pre, tonesTiming], axis=1)

# define ylims here
ylims = {'s1': [0, 7],
         's2RT': [-300, 500],
         's2acc': [0.5, 1]
}

# call function to plot
kwargs = {'evbounds': dfDes_peaks_c}
dualplot_tonesWevents(evntBound, meanRT, meanAcc, ylims, importPaths, **kwargs)
#dualplot_tonesWevents(evntBound, abc_final_c1, meanAcc, ylims, **kwargs)
plt.show()


'''
# event boundaries from orig dataset ==========================================
countsEvtB = [i.sum(axis = 0) for i in init_es.allDat['aEventBounds']]
meanCtEvtB_c1 = aCountEvtB[stup['c_inds'][0]].mean()
meanCtEvtB_c2 = aCountEvtB[stup['c_inds'][1]].mean()
'''
