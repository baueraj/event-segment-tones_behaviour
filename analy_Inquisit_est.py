from init_est import *
from analy_funs_est import *
sys.path.append('/Users/bauera/Dropbox/UofT/experiments/event-segmentation/analysis')
import init_es
#plt.close('all')
plt.show()

# other vars
stup = {'smooth_win': 2
}

# pilot 2 =====================================================================
allDat_plt2_pre = get_participant_data_plt2(paths_plt2, cartoonNames, propTrialsThresh)
allDat_plt2 = prep_subj_data_plt2(allDat_plt2_pre, paths_plt2)

# plot tone data on top of adults' event boundaries (just rugrats clip 1)
dfDesign = pd.read_csv(paths_plt2[1])
i_c = 0
inds_c = [i_c*2, i_c*2 + 1]
dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                          .dropna(axis=0, how='any')
tonesTiming = dfDesign_c['bigPeople_rugrats_1_onset'].copy()/1000
tonesTiming.name = 'onset'

#aSmthSumEvtB = []
#for i in [1]:
#    aSmthSumEvtB.append(init_es.plot_mov_avg_events(init_es.allDat['aEventBounds'][i].sum(axis=1),
#                                                    stup['smooth_win'], 'b', 1, 1, 1).fillna(value=0))
evntBound = init_es.plot_mov_avg_events(init_es.allDat['aEventBounds'][i].sum(axis=1),
                                        stup['smooth_win'], 'b', 1, 0, 1).fillna(value=0)

meanRT_pre = allDat_plt2['RT'][0].mean(axis=1)
meanRT_pre.name = 'RT'
meanRT = pd.concat([meanRT_pre, tonesTiming], axis=1)

meanAcc_pre = allDat_plt2['acc'][0].mean(axis=1)
meanAcc_pre.name = 'acc'
meanAcc = pd.concat([meanAcc_pre, tonesTiming], axis=1)

ylims = {'s1': [0, 7],
         's2RT': [650, 1150],
         's2acc': [0.5, 1]
}

dualplot_tonesWevents(evntBound, meanRT, meanAcc, ylims)

