import os, sys, pdb
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

"""
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)
"""

cartoonNames = ['bigPeople_rugrats', 'treasureHunt_busyWorld']
#hthreshold = 3000 #ms

# pilot 1 =====================================================================
dPath_plt1 = '../data/plt1/main_tonescartoons_raw_17_10_12.csv'
designPath_plt1 = '../plt1/videos_toneOnsetsNIDs.csv'
paths_plt1 = [dPath_plt1, designPath_plt1]
# before 9/15 date (i.e. < 91517)
vid_selector = {1: 'c1_1',
                2: 'c2_1',
                3: 'c1_2',
                4: 'c2_2',
                5: 'c1_3'}
# 9/15 date (i.e. == 91517)
#vid_select_mod6eq0 = 'c2_3'

# pilot 2 (start: 10/17/17) ===================================================
#dPath_plt2_c1_clip1 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid1_raw_17_10_20.csv']]
dPath_plt2_c1_clip1 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid1_raw_17_10_26.csv']]
dPaths_plt2 = [dPath_plt2_c1_clip1]

designPath_plt2 = '../designMaterials/plt2/videos_toneOnsetsNIDs.csv'

paths_plt2 = [dPaths_plt2, designPath_plt2]

propTrialsThresh = 0.9
