import json
import sys
import pandas as pd
import time
import rasterio
import numpy as np
import shutil
import os
from os import listdir

def compare_counts(sourcedir , comparedir , domain):
    all_source_folders = listdir(sourcedir)
    cnt_source = 0
    for folder in all_source_folders:
        cnt_source = cnt_source + 1

    cnt_target = 0
    all_compare_folders = listdir(comparedir)
    for folder in all_compare_folders:
        foldername = comparedir + '/' + folder + '/dataset/' + domain
        all_files = listdir(foldername)
        for filename in all_files:
            cnt_target = cnt_target + 1
    return cnt_source , cnt_target       

sourcepath = sys.argv[1]
sourcedir = sourcepath + '/CUT/dataset/trainA'
comparedir = sourcepath + '/CUTdistance/train/'
domain = 'trainA'
cnt_source , cnt_target = compare_counts(sourcedir , comparedir , domain)
print ('Total images in CUT train trainA source:' , cnt_source, 'Total images in CUT train trainA split:' , cnt_target)

sourcedir = sourcepath + '/CUT/dataset/trainB'
comparedir = sourcepath + '/CUTdistance/train/'
domain = 'trainB'
cnt_source , cnt_target = compare_counts(sourcedir , comparedir , domain)
print ('Total images in CUT train trainB source:' , cnt_source, 'Total images in CUT train trainB split:' , cnt_target)


sourcedir = sourcepath + '/CUT/predict/dataset/trainA'
comparedir = sourcepath + '/CUTdistance/predict/'
domain = 'trainA'
cnt_source , cnt_target = compare_counts(sourcedir , comparedir , domain)
print ('Total images in CUT predict trainA source:' , cnt_source, 'Total images in CUT predict trainA split:' , cnt_target)

sourcedir = sourcepath + '/CUT/predict/dataset/trainB'
comparedir = sourcepath + '/CUTdistance/predict/'
domain = 'trainB'
cnt_source , cnt_target = compare_counts(sourcedir , comparedir , domain)
print ('Total images in CUT predict trainA source:' , cnt_source, 'Total images in CUT predict trainA split:' , cnt_target)


