dfs_analy_byC['acc'][0].mean(axis=0)
"""
12    0.945736
16    0.635659
26    0.798450
45    0.395349
46    0.798450
"""

dfs_analy_byC['RT'][0].drop([45, 46]).mean(axis=1)