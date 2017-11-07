def get_participant_data(paths, cartoonNames, propTrialsThresh):
    '''
    read participant data stored as single csv file

    Parameters
    ----------
    paths : list
        string paths to files, folders
    cartoonNames : list
        strings of cartoon names
    propTrialsThresh : float
        specifies proportion of trials that must have a response to include subj data 
        
    Returns
    -------
    dfs_byC : list
        contains lists by cartoon clip of each subj data df

    Notes
    -----
    NA
    '''

    import pdb
    import numpy as np
    import pandas as pd
    
    dPaths = paths[0]
    dfDesign = pd.read_csv(paths[1])

    columns_to_retain = ['subject', 'trialcode', 'trialnum', 'values.key_pressed', 
                         'values.timeOfResponse', 'block.video_play.timestamp', 
                         'trial.Response.timestamp', 'video.video_in.timestamp']
    
    dfs_byC = []
    
    for i_c, dPathsAll_c in enumerate(dPaths):
        
        inds_c = [i_c*2, i_c*2 + 1]
        dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                                  .dropna(axis=0, how='any')

        for j_c, dPath_c in enumerate(dPathsAll_c):
            
            df_all = pd.read_csv(dPath_c)[columns_to_retain]
            df = df_all[(df_all['trialcode'] == 'write_videodata_trial')].copy()

            dfs_bySubj_pre = [df[(df['subject'] == i)].reset_index() for i in df['subject'].unique() \
                              if len(df[(df['subject'] == i)]) / len(dfDesign_c) >= propTrialsThresh]

            dfs_bySubj = [pd.concat([df_i, pd.DataFrame({'elapTime': df_i['values.timeOfResponse'] - (df_i.loc[0,'video.video_in.timestamp'] - \
                                                        df_i.loc[0,'block.video_play.timestamp'])})], axis=1) for df_i in dfs_bySubj_pre]
            
            if j_c == 0:
                dfs_bySubjAll = dfs_bySubj
            else:
                dfs_bySubjAll.extend(dfs_bySubj)
                
        dfs_byC.append(dfs_bySubjAll)
            
    #pdb.set_trace()
    
    return dfs_byC



def prep_subj_data(dfs_dat_byC, cs_withDat, paths):
    '''
    detrend and reformat participant data

    Parameters
    ----------
    dfs_dat_byC : list
        contains lists by cartoon clip of each subj data df
    cs_withDat : list
        specifies indices of cartoon clips that have data to prep (inds based on design file)
    paths : list
        string paths to files, folders
    
    Returns
    -------
    dfs_analy_byC : dict
        contains lists by cartoon clip of RT and acc dfs

    Notes
    -----
    Assumes for now low tone (0) correct response is '36', high (1) is '37'
    '''

    import pdb
    import numpy as np
    import pandas as pd
    from scipy import signal
    
    dfDesign = pd.read_csv(paths[1])
    
    dfs_RT_byC = []
    dfs_corrResp_byC = []
    
    wrngVal = 0
    
    for iDat_c, dfs_c in enumerate(dfs_dat_byC):
        i_c = cs_withDat[iDat_c]
        
        inds_c = [i_c*2, i_c*2 + 1]
        dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                                  .dropna(axis=0, how='any')

        df_RT_c = pd.DataFrame()
        df_corrResp_c = pd.DataFrame()
        
        for i in range(len(dfDesign_c)):
            
            if dfDesign_c.iloc[i, 0] == 0:
                corrKey = 36 #low tone
            else:
                corrKey = 37 #high tone
            toneOnset = dfDesign_c.iloc[i, 1]
            
            subjs_i = []
            RT_i = []
            corrResp_i = []
            
            for df_subj in dfs_c:
            
                subjs_i.append(df_subj.loc[0, 'subject'])
                trialInd = pd.Series.idxmin(df_subj['elapTime'].where(df_subj['elapTime'] > toneOnset))
                corrResp_i.append(1 if df_subj.loc[trialInd, 'values.key_pressed'] == corrKey else wrngVal)                
                RT_i.append(df_subj.loc[trialInd, 'elapTime'] - toneOnset)

                if corrResp_i[-1] == wrngVal:
                    RT_i[-1] = np.nan
                        
                if i != len(dfDesign_c)-1:
                    #pdb.set_trace()
                    if df_subj.loc[trialInd, 'elapTime'] > dfDesign_c.iloc[i+1, 1]:
                        corrResp_i[-1] = wrngVal
                        RT_i[-1] = np.nan
              
            df_RT_c = df_RT_c.append(dict(zip(subjs_i, RT_i)), ignore_index=True)
            df_corrResp_c = df_corrResp_c.append(dict(zip(subjs_i, corrResp_i)), ignore_index=True)
        
        # to detrend, I need to ignore nan values [can't just use df_RT_c.transform(lambda x: signal.detrend(x))]
        for col_i in list(df_RT_c.columns):
            col_pre = df_RT_c[col_i][df_RT_c[col_i].notnull()].copy()
            col_detrend = signal.detrend(col_pre)
            df_RT_c.loc[col_pre.index, col_i] = col_detrend
            
        dfs_RT_byC.append(df_RT_c)
        dfs_corrResp_byC.append(df_corrResp_c)
        
    dfs_analy_byC = {'RT': dfs_RT_byC,
                     'acc': dfs_corrResp_byC}
                
    return dfs_analy_byC



def prep_tone_timestamps(paths):
    '''
    prepare each cartoon clip's tones for LMER (distance from event boundary, etc.)

    Parameters
    ----------
    paths : list
        contains paths to design materials
    
    Returns
    -------
    dfs_tones : list
        contains df per cartoon clip of vars of interest for LMER

    Notes
    -----
    NA
    '''
    
    import pdb
    import numpy as np
    import pandas as pd
    
    stup = {'peak_win': 2,
            'after_peak_rng': [2, 5],
            'exclude_dist': 10}
    
    dfDesign = pd.read_csv(paths[1])
    dfDesign_peaks = pd.read_csv(paths[2])
    
    dfs_tones = []
    
    for i_c in range(int(len(dfDesign.columns)/2)):        
        ind_c = i_c*2 + 1
        
        dfDesign_c = dfDesign.iloc[:, ind_c].copy() \
                                  .dropna(axis=0, how='any')
 
        dfDes_peaks_c = dfDesign_peaks.iloc[:, int(np.floor(i_c/(len(dfDesign.columns)/2)))].copy() \
                                  .dropna(axis=0, how='any')
        
        dist_c = []
        peak_c = []
        after_peak_c = []
        before_frst_bound_c = []
        
        for j, tone_j in enumerate(dfDesign_c): #it's a series -- need to specify name?
            past_peak_ind = pd.Series.idxmax(dfDes_peaks_c.where(dfDes_peaks_c < tone_j/1000))
            future_peak_ind = pd.Series.idxmin(dfDes_peaks_c.where(dfDes_peaks_c > tone_j/1000))
            
            dist_c.append(tone_j/1000 - dfDes_peaks_c.loc[past_peak_ind]) # NOTE: loc, not iloc -- pd.Series.idxmax returns index
            if dist_c[-1] > stup['exclude_dist']: dist_c[-1] = np.nan
                     
            if (tone_j/1000 - dfDes_peaks_c.loc[past_peak_ind] <= stup['peak_win'] or \
                                          dfDes_peaks_c.loc[future_peak_ind] - tone_j/1000 <= stup['peak_win']) and \
                                          dfDes_peaks_c.loc[past_peak_ind] != 0:
                peak_c.append(1)
            else:
                peak_c.append(0)
                
            if tone_j/1000 - dfDes_peaks_c.loc[past_peak_ind] >= stup['after_peak_rng'][0] and \
                                          tone_j/1000 - dfDes_peaks_c.loc[past_peak_ind] <= stup['after_peak_rng'][1] and \
                                          dfDes_peaks_c.loc[past_peak_ind] != 0:
                after_peak_c.append(1)
            else:
                after_peak_c.append(0)   
                  
            before_frst_bound_c.append(1 if dfDes_peaks_c.loc[past_peak_ind]==0 else 0)
        
        dfTones_c = pd.DataFrame({'dist': dist_c, 'peak': peak_c, 'afterPeak': after_peak_c, 'b41stBound': before_frst_bound_c})
        dfTones_c.loc[dfTones_c.isnull().any(axis=1), :] = np.nan
        dfs_tones.append(dfTones_c)
        
    return dfs_tones



def dualplot_tonesWevents(s1, s2RT, s2acc, ylims):
    '''
    plots tone data with event boundaries from original event seg data

    Parameters
    ----------
    s1 : pandas series/df (1 col)
        moothed event boundary data to be plotted
    s2RT : pandas df
        tones data (RT) with tone 'onset' column
    s2acc : pandas df
        tones data (acc) with tone 'onset' column
    ylims : dict
        ylims for plotting (each is a list)

    Returns
    -------
    None

    Notes
    -----
    Not currently plotting accuracy data
    '''

    import os, sys, pdb
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    sys.path.append('/Users/bauera/Dropbox/UofT/experiments/event-segmentation/analysis')
    import init_es
    plt.show()
    
    # using 20 seconds as a non-parameterized (in function) default increment for x-ticks
    stepSize = 20

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    s1.plot.area(ax=ax, color='b', alpha=1)
    s2RT.plot.line(x='onset', y='RT', ax=ax2, linewidth=3, style='r-o', secondary_y=True)
    
    rng = np.arange(0, max(s1.index) + stepSize, stepSize)
    ax.set_xticks(rng)
    ax.set_xticklabels(init_es.reformat_timestamp(rng).values, rotation=-45)
    
    ax.set_ylim(ylims['s1'])
    ax2.right_ax.set_ylim(ylims['s2RT'])


# STOPPED
"""
def get_n_peak_bounds(stup_es):
    '''
    returns and plots n peak event boundaries (Zacks definition) for rugrats and busyTown

    Parameters
    ----------
    xxx
    
    Returns
    -------
    xxx : list
        timestamps of the n peak event boundaries (at their peaks)

    Notes
    -----
    NA
    '''

    import os, sys, pdb
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    sys.path.append('/Users/bauera/Dropbox/UofT/experiments/event-segmentation/analysis')
    import init_es
    plt.show()
    
    countsEvtB = [i.sum(axis = 0) for i in init_es.allDat['aEventBounds']]
    meanCtEvtB_c1 = aCountEvtB[stup['c_inds'][0]].mean()
    meanCtEvtB_c2 = aCountEvtB[stup['c_inds'][1]].mean()
    
    aSmthSumEvtB = []
    for i in stup['c_inds']:
        allEvtB = init_es.plot_mov_avg_events(init_es.allDat['aEventBounds'][i].sum(axis=1),
                                              stup['smooth_win'], 'b', 1, 0, 1).fillna(value=0)
    
    return
"""



# OLD
"""
def get_participant_data_plt2(paths, cartoonNames, propTrialsThresh):
    '''
    read participant data stored as single csv file

    Parameters
    ----------
    paths : list
        string paths to files, folders
    cartoonNames : list
        strings of cartoon names
    propTrialsThresh : float
        specifies proportion of trials that must have a response to include subj data 
        
    Returns
    -------
    dfs_byC : list
        contains lists by cartoon clip of each subj data df

    Notes
    -----
    NA
    '''
    
    import pdb
    import numpy as np
    import pandas as pd
    
    dPaths = paths[0]
    dfDesign = pd.read_csv(paths[1])

    columns_to_retain = ['blockcode', 'subject', 'trialnum', 'response', 'latency']
    
    dfs_byC = []
    
    for i_c, dPathsAll_c in enumerate(dPaths):
        
        inds_c = [i_c*2, i_c*2 + 1]
        dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                                  .dropna(axis=0, how='any')

        for j_c, dPath_c in enumerate(dPathsAll_c):
            
            df_all = pd.read_csv(dPath_c)[columns_to_retain]
            df = df_all[(df_all['blockcode'] == 'video_play')].copy()

            dfs_bySubj_pre = [df[(df['subject'] == i)].reset_index() for i in df['subject'].unique() \
                               if len(df[(df['subject'] == i)]) / len(dfDesign_c) >= propTrialsThresh]
            
            dfs_bySubj = [pd.concat([df_i, pd.DataFrame({'elapTime': df_i['latency'].cumsum(axis=0)})], axis=1) \
                          for df_i in dfs_bySubj_pre]
            
            if j_c == 0:
                dfs_bySubjAll = dfs_bySubj
            else:
                dfs_bySubjAll.extend(dfs_bySubj)
                
        dfs_byC.append(dfs_bySubjAll)
            
    #pdb.set_trace()
    
    return dfs_byC
"""



# OLD
"""
def prep_subj_data_plt2(dfs_dat_byC, designPath):
    '''
    reformat participant data

    Parameters
    ----------
    dfs_dat_byC : list
        contains lists by cartoon clip of each subj data df
    designPath : string
        specifies path to exp design csv
    
    Returns
    -------
    dfs_analy_byC : dict
        contains lists by cartoon clip of RT and acc dfs

    Notes
    -----
    Assumes for now low tone (0) correct response is '36', high (1) is '37'
    Hard-coded lowerbound for acceptable RT (100ms)
    '''    
    import pdb
    import numpy as np
    import pandas as pd
    
    dfDesign = pd.read_csv(paths[1])
    
    dfs_RT_byC = []
    dfs_corrResp_byC = []
    
    wrngVal = 0
    
    for i_c, dfs_c in enumerate(dfs_dat_byC):
        
        inds_c = [i_c*2, i_c*2 + 1]
        dfDesign_c = dfDesign.iloc[:, inds_c].copy() \
                                  .dropna(axis=0, how='any')
        
        df_RT_c = pd.DataFrame()
        df_corrResp_c = pd.DataFrame()
        
        for i in range(len(dfDesign_c)):
            
            if dfDesign_c.iloc[i, 0] == 0:
                corrKey = '36' #low tone
            else:
                corrKey = '37' #high tone
            toneOnset = dfDesign_c.iloc[i, 1]
            
            subjs_i = []
            RT_i = []
            corrResp_i = []   
            
            for df_subj in dfs_c:
            
                subjs_i.append(df_subj.loc[0, 'subject'])
                trialInd = np.argmin(df_subj['elapTime'].where(df_subj['elapTime'] > toneOnset))
                corrResp_i.append(1 if df_subj.loc[trialInd, 'response'] == corrKey else wrngVal)                
                RT_i.append(df_subj.loc[trialInd, 'elapTime'] - toneOnset)
                
                if corrResp_i[-1] == wrngVal:
                    RT_i[-1] = np.nan
                        
                if RT_i[-1] <= 100:
                    corrResp_i[-1] = wrngVal
                    RT_i[-1] = np.nan
                        
                if i != len(dfDesign)-1:   
                    if df_subj.loc[trialInd, 'elapTime'] > dfDesign_c.iloc[i+1, 1]:
                        corrResp_i[-1] = wrngVal
                        RT_i[-1] = np.nan
              
            df_RT_c = df_RT_c.append(dict(zip(subjs_i, RT_i)), ignore_index=True)
            df_corrResp_c = df_corrResp_c.append(dict(zip(subjs_i, corrResp_i)), ignore_index=True)
            
        dfs_RT_byC.append(df_RT_c)
        dfs_corrResp_byC.append(df_corrResp_c)
        
    dfs_analy_byC = {'RT': dfs_RT_byC,
                     'acc': dfs_corrResp_byC}
                
    return dfs_analy_byC
"""


# OLD
"""
def get_participant_data_plt1(vid_selector, paths):
    '''
    read participant data stored as single csv file

    Parameters
    ----------
    vid_selector : dictionary
        mod 6 values to video names
    paths : list
        string paths to files, folders
        
    Returns
    -------
    allDat : dictionary (of data)

    Notes
    -----
    UNFINISHED
    '''
    
    import pdb
    import numpy as np
    import pandas as pd
    
    dPath = paths[0]
    designPath = paths[1]

    columns_to_retain = ['date', 'blockcode', 'subject', 'trialnum', 'response', 'latency']

    df_all = pd.read_csv(dPath)
    df = df_all[columns_to_retain]
    #df.reset_index(drop=True, inplace=True)
    df_mod6eq0 = df[(df['blockcode'] == 'video_play') & (df['date'] == 91517)].copy()
    df = df[(df['blockcode'] == 'video_play') & (df['date'] < 91517)]
    
    # cartoon IDs mod % 6 == 1 - 5
    c_wSubj_temp = [df[(df['subject'] % 6 == i)] for i in list(vid_selector.keys())]
    
    c_wSubj = []
    for i in range(0, len(c_wSubj_temp)):
        c_wSubj.append([c_wSubj_temp[i][(c_wSubj_temp[i]['subject'] == j)] \
                        for j in c_wSubj_temp[i]['subject'].unique()])
    
    # cartoon ID mod % 6 == 0
    c_wSubj.append([df_mod6eq0[(df_mod6eq0['subject'] == j)] \
                        for j in df_mod6eq0['subject'].unique()])
    
    allDat = {}
    
    return allDat
"""
