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



def prep_subj_data(dfs_dat_byC, designPath):
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
