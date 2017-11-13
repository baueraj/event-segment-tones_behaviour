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
        
        # to detrend, must ignore nan values [can't just use df_RT_c.transform(lambda x: signal.detrend(x))]
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
            'exclude_dist': 15}
    
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
        
        for j, tone_j in enumerate(dfDesign_c):
            
            past_peak_ind = pd.Series.idxmax(dfDes_peaks_c.where(dfDes_peaks_c < tone_j/1000))
            if not np.isnan(past_peak_ind):
                past_peak = dfDes_peaks_c.loc[past_peak_ind]
            else:
                past_peak = np.nan
            
            future_peak_ind = pd.Series.idxmin(dfDes_peaks_c.where(dfDes_peaks_c > tone_j/1000))
            if not np.isnan(future_peak_ind):
                future_peak = dfDes_peaks_c.loc[future_peak_ind]
            else:
                future_peak = np.nan
            
            dist_c.append(tone_j/1000 - past_peak)
            if dist_c[-1] > stup['exclude_dist']: dist_c[-1] = np.nan

            if (tone_j/1000 - past_peak <= stup['peak_win'] or \
                                          future_peak - tone_j/1000 <= stup['peak_win']) and \
                                          past_peak != 0:
                peak_c.append(1)
            else:
                peak_c.append(0)
                
            if tone_j/1000 - past_peak >= stup['after_peak_rng'][0] and \
                                          tone_j/1000 - past_peak <= stup['after_peak_rng'][1] and \
                                          past_peak != 0:
                after_peak_c.append(1)
            else:
                after_peak_c.append(0)   
                  
            before_frst_bound_c.append(1 if past_peak==0 else 0)
        
        dfTones_c = pd.DataFrame({'dist': dist_c, 'peak': peak_c, 'afterPeak': after_peak_c, 'before1stBound': before_frst_bound_c})
        dfTones_c.loc[dfTones_c.isnull().any(axis=1), :] = np.nan
        dfs_tones.append(dfTones_c)
        
    return dfs_tones



def dualplot_tonesWevents(s1, s2RT, s2acc, ylims, importPaths, **kwargs):
    '''
    plots tone data with event boundaries from original event seg data

    Parameters
    ----------
    s1 : pandas series/df (1 col)
        smoothed event boundary data to be plotted
    s2RT : pandas df
        tones data (RT) with tone 'onset' column
    s2acc : pandas df
        tones data (acc) with tone 'onset' column
    ylims : dict
        ylims for plotting (each is a list)
    importPaths : list
        contains paths for importing other modules
    kwargs
        evbounds : pandas series
            contains timestamps of event boundaries to plot

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
    from matplotlib import style
    style.use('ggplot')
    plt.show()
    for imPath_i in importPaths:
        sys.path.append(imPath_i)
    import init_es
    
    # using 20 seconds as a non-parameterized default increment for x-ticks
    stepSize = 20

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    s1.plot.area(ax=ax, color='b', alpha=1)
    s2RT.plot.line(x='onset', y='RT', ax=ax2, linewidth=3, style='r-o', secondary_y=True)
    ax2.legend_.remove() 
    
    #rng = np.arange(0, max(s1.index) + stepSize, stepSize)
    #ax.set_xticks(rng)
    rng = ax.get_xticks()
    ax.set_xticklabels(init_es.reformat_timestamp(rng).values, rotation=-45)
    ax.set_xlabel('Time')
    ax.xaxis.label.set_size(17)
    ax.set_ylabel('Event boundary vote')
    ax.yaxis.label.set_size(17)
    ax.yaxis.label.set_color('blue')
    ax.set_ylim(ylims['s1'])    
    ax.grid(False)
    
    ax2.right_ax.set_ylim(ylims['s2RT'])
    ax2.right_ax.set_ylabel('Reaction time to tone')
    ax2.right_ax.yaxis.label.set_size(17)
    ax2.right_ax.yaxis.label.set_color('red')
    ax2.right_ax.grid(False)
    
    for xc in kwargs['evbounds']:
       plt.axvline(x=xc, color='k', linestyle='--', linewidth=1.5)



def get_rt_bound_waveforms(dfs_analy_byC, paths):
    '''
    plot mean waveforms around event boundaries

    Parameters
    ----------
    dfs_analy_byC : dict
        contains lists by cartoon clip of RT and acc dfs 
    paths : list
        contains paths to design materials
    
    Returns
    -------
    None

    Notes
    -----
    Assumes have data for all cartoons
    Work in progress
    '''

    import pdb
    import numpy as np
    import pandas as pd

    dfs_RT = dfs_analy_byC['RT'].copy()

    dfDesign = pd.read_csv(paths[1])
    dfDesign_peaks = pd.read_csv(paths[2])

    c1_inds = [0, 2, 4]
    c2_inds = [1, 3, 5]

    dfDes_peaks_c1 = dfDesign_peaks.iloc[:, 0].copy() \
                                  .dropna(axis=0, how='any')

    dfDes_peaks_c2 = dfDesign_peaks.iloc[:, 1].copy() \
                                  .dropna(axis=0, how='any')

    win_peak_pastfut = 15000 # ms                     
    
    # cartoon 1 ===============================================================    
    collect_dfs_RT = []
    
    for idx, i_c in enumerate(c1_inds):
    
        dfDesign_c = dfDesign.iloc[:, i_c*2 + 1].copy() \
                                  .dropna(axis=0, how='any')

        dfDes_peaks_c = dfDesign_peaks.iloc[:, int(np.floor(i_c/(len(dfDesign.columns)/2)))].copy() \
                                  .dropna(axis=0, how='any') 

        df = dfs_RT[i_c].copy()
        dfDesign_c.name = 'toneOnset'
        df.index = dfDesign_c
        
        collect_dfs_RT.append(df.mean(axis=1))
    
    a=collect_dfs_RT[0]
    b=collect_dfs_RT[1]
    c=collect_dfs_RT[2]
    
    aind=a.index
    bind=b.index
    cind=c.index
    
    abcind = list(set(aind) & set(bind) & set(cind))
    abind = list(set(aind) & set(bind))
    abind = [x for x in abind if x not in abcind]
    acind = list(set(aind) & set(cind))
    acind = [x for x in acind if x not in abcind]
    bcind = list(set(bind) & set(cind))
    bcind = [x for x in bcind if x not in abcind]
    
    abc = pd.concat([a.loc[abcind], b.loc[abcind], c.loc[abcind]], axis=1).mean(axis=1)
    ab = pd.concat([a.loc[abind], b.loc[abind]], axis=1).mean(axis=1)
    ac = pd.concat([a.loc[acind], c.loc[acind]], axis=1).mean(axis=1)
    bc = pd.concat([b.loc[bcind], c.loc[bcind]], axis=1).mean(axis=1)
    
    a.drop(abcind, inplace=True)
    b.drop(abcind, inplace=True)
    c.drop(abcind, inplace=True)
    a.drop(abind, inplace=True, errors='ignore')
    b.drop(abind, inplace=True, errors='ignore')
    a.drop(acind, inplace=True, errors='ignore')
    c.drop(acind, inplace=True, errors='ignore')
    b.drop(bcind, inplace=True, errors='ignore')
    c.drop(bcind, inplace=True, errors='ignore')
    
    abc_final_c1 = pd.concat([a,b,c,abc,ab,ac,bc]).sort_index(ascending=True)
    
    # get individual waveforms around boundaries, and mean
    
    collect_waves = []
    
    for i in (dfDes_peaks_c1.loc[1:] * 1000):
        tones_win_mask = np.abs(abc_final_c1.index - i) <= win_peak_pastfut
        tones_masked = abc_final_c1[tones_win_mask]
        tones_masked_index = pd.Series(abc_final_c1.index - i)[tones_win_mask]
        tones_masked.index = pd.to_datetime(tones_masked_index, unit='ms')
        collect_waves.append(tones_masked)
    
    allWaves = pd.concat(collect_waves).sort_index(ascending=True)
    plt.plot(allWaves.rolling('1000ms').mean())
    plt.plot(allWaves)

    '''    
    for i in (dfDes_peaks_c1.loc[1:] * 1000):
        tones_win_mask = np.abs(a.index - i) <= 5
        tones_masked = a[tones_win_mask]
        tones_masked_index = pd.Series(a.index - i)[tones_win_mask]
        tones_masked.index = pd.to_datetime(tones_masked_index, unit='ms')
        collect_waves.append(tones_masked)    
        
    allWaves = collect_waves[1]
    allWaves.rolling('1s').mean()
    plt.plot(allWaves)
    '''
    
    # cartoon 2 ===============================================================    
    collect_dfs_RT = []
    
    for idx, i_c in enumerate(c2_inds):
    
        dfDesign_c = dfDesign.iloc[:, i_c*2 + 1].copy() \
                                  .dropna(axis=0, how='any')

        df = dfs_RT[i_c].copy()
        dfDesign_c.name = 'toneOnset'
        df.index = dfDesign_c
        
        collect_dfs_RT.append(df.mean(axis=1))
    
    a=collect_dfs_RT[0]
    b=collect_dfs_RT[1]
    c=collect_dfs_RT[2]
    
    aind=a.index
    bind=b.index
    cind=c.index
    
    abcind = list(set(aind) & set(bind) & set(cind))
    abind = list(set(aind) & set(bind))
    abind = [x for x in abind if x not in abcind]
    acind = list(set(aind) & set(cind))
    acind = [x for x in acind if x not in abcind]
    bcind = list(set(bind) & set(cind))
    bcind = [x for x in bcind if x not in abcind]
    
    abc = pd.concat([a.loc[abcind], b.loc[abcind], c.loc[abcind]], axis=1).mean(axis=1)
    ab = pd.concat([a.loc[abind], b.loc[abind]], axis=1).mean(axis=1)
    ac = pd.concat([a.loc[acind], c.loc[acind]], axis=1).mean(axis=1)
    bc = pd.concat([b.loc[bcind], c.loc[bcind]], axis=1).mean(axis=1)
    
    a.drop(abcind, inplace=True)
    b.drop(abcind, inplace=True)
    c.drop(abcind, inplace=True)
    a.drop(abind, inplace=True, errors='ignore')
    b.drop(abind, inplace=True, errors='ignore')
    a.drop(acind, inplace=True, errors='ignore')
    c.drop(acind, inplace=True, errors='ignore')
    b.drop(bcind, inplace=True, errors='ignore')
    c.drop(bcind, inplace=True, errors='ignore')
    
    abc_final_c2 = pd.concat([a,b,c,abc,ab,ac,bc]).sort_index(ascending=True)
    
    # get individual waveforms around boundaries, and mean
    
    collect_waves = []
    
    for i in (dfDes_peaks_c2.loc[1:] * 1000):
        tones_win_mask = np.abs(abc_final_c2.index - i) <= win_peak_pastfut
        tones_masked = abc_final_c2[tones_win_mask]
        tones_masked_index = pd.Series(abc_final_c2.index - i)[tones_win_mask]
        tones_masked.index = pd.to_datetime(tones_masked_index, unit='ms')
        collect_waves.append(tones_masked)
    
    allWaves = pd.concat(collect_waves).sort_index(ascending=True)
    plt.plot(allWaves.rolling('1000ms').mean())
    plt.plot(allWaves)
