import json
import sys
import pandas as pd
import time
import rasterio
import numpy as np
import shutil
import os
from os import listdir

def copy_files(dirpath,top_folder,sub_folder,path_array):
    all_folders = listdir(dirpath)
    for filename in all_folders:
        file_with_path = sourcepath + '/' + filename.replace('.jpg','')  + '/' + filename.replace('.jpg','')  + '_labels_metadata.json'
        with open(file_with_path) as f:
            coordinate = data = float(json.load(f)['coordinates']['uly'])/1000
            if coordinate > 4000 and coordinate < 5000:
                savepath = writepath + '/' + top_folder  + '/' + path_array[0] + '/dataset/' + sub_folder + '/'  + filename
            elif coordinate >= 5000 and coordinate < 6000:
                savepath = writepath + '/' + top_folder  + '/' + path_array[1] + '/dataset/' + sub_folder + '/' + filename
        #elif coordinate >= 6000 and coordinate < 7000:
        #    savepath = writepath + '/train/6K/dataset/trainA/' + filename
            else:
                savepath = writepath + '/' + top_folder  + '/' + path_array[2]+ '/dataset/' + sub_folder + '/' + filename
            sourcefile = dirpath + '/' + filename
            shutil.copyfile(sourcefile,savepath)

path_array = ['4K','5K' , '6K' ]
outputpath=sys.argv[2]
sourcepath = sys.argv[1]
writepath= outputpath + '/' + 'CUTdistance'
if os.path.exists(writepath):
    shutil.rmtree(writepath)
os.mkdir(writepath)
predictpath = outputpath + '/' + 'CUTdistance/predict'
os.mkdir(predictpath)
trainpath = outputpath + '/' + 'CUTdistance/train'
os.mkdir(trainpath)
for pathname in path_array:
    foldername = predictpath + '/' + pathname
    os.mkdir(foldername)
    foldername = foldername + '/dataset'
    os.mkdir(foldername)
    trainA = foldername + '/trainA'
    os.mkdir(trainA)
    trainB = foldername + '/trainB'
    os.mkdir(trainB)

    foldername = trainpath + '/' + pathname
    os.mkdir(foldername)
    foldername = foldername + '/dataset'
    os.mkdir(foldername)
    trainA = foldername + '/trainA'
    os.mkdir(trainA)
    trainB = foldername + '/trainB'
    os.mkdir(trainB)


top_folder = 'train'
sub_folder = 'trainA'
dirpath = outputpath + '/CUT/dataset/trainA'
copy_files(dirpath,top_folder,sub_folder,path_array)
sub_folder = 'trainB'
dirpath = outputpath + '/CUT/dataset/trainB'
copy_files(dirpath,top_folder,sub_folder,path_array)

top_folder = 'predict'
sub_folder = 'trainA'
dirpath = outputpath + '/CUT/predict/dataset/trainA'
copy_files(dirpath,top_folder,sub_folder,path_array)
sub_folder = 'trainB'
dirpath = outputpath + '/CUT/predict/dataset/trainB'
copy_files(dirpath,top_folder,sub_folder,path_array)

