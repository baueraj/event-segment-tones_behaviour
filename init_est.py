import os, sys, pdb
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



"""
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
"""



# pilot 2 (start: 10/17/17) ===================================================
dPath_plt2_c1_clip1 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid1_raw_17_10_26.csv']]
dPath_plt2_c1_clip2 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid3_raw_17_11_05.csv']]
dPath_plt2_c1_clip3 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid5_raw_17_11_03.csv']]

dPath_plt2_c2_clip1 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid2_raw_17_11_06.csv']]
dPath_plt2_c2_clip2 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid4_raw_17_11_06.csv']]
dPath_plt2_c2_clip3 = ['../data/plt2/' + i for i in ['main_tonescartoons_vid6_raw_17_11_06.csv']]

dPaths_plt2 = [dPath_plt2_c1_clip1, dPath_plt2_c2_clip1, 
               dPath_plt2_c1_clip2, dPath_plt2_c2_clip2, 
               dPath_plt2_c1_clip3, dPath_plt2_c2_clip3]
cs_withDat_plt2 = [0, 1, 2, 3, 4, 5] # cartoon clips with data

designPath_plt2 = '../designMaterials/plt2/videos_toneOnsetsNIDs.csv'
designPath2_plt2 = '../designMaterials/plt2/n_peak_boundaries_rollAvg.csv'
paths_plt2 = [dPaths_plt2, designPath_plt2, designPath2_plt2]

propTrialsThresh = 0.95



# other vars ==================================================================
cartoonNames = ['bigPeople_rugrats', 'treasureHunt_busyWorld']
importPath_1 = '/Users/bauera/Dropbox/UofT/experiments/event-segmentation/analysis'
importPaths = [importPath_1]
