from os import listdir
import json
import sys
import pandas as pd
import time
import rasterio
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import math
import random
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def visualizeGeoData(imagepath,writepath,image_folder):
    # Initialize subplots
    #fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 20), sharey=True)
    fp2 = imagepath + "_B02.tif"
    fp3 = imagepath + "_B03.tif"
    fp4 = imagepath + "_B04.tif"
    #fp8 = imagepath + "_B08.tif"

    raster2= rasterio.open(fp2)
    raster3 = rasterio.open(fp3)
    raster4 = rasterio.open(fp4)
    #raster8 = rasterio.open(fp8)
        #print("Coordinate reference su=ystem ",raster2.crs)
        # Read the grid values into numpy arrays
    red = raster4.read(1)
    green = raster3.read(1)
    blue = raster2.read(1)
    #nir = raster8.read(1)
        # Normalize the bands
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)
    #nirn = normalize(nir)
        # Create RGB natural color composite
    #rgbnir = np.dstack((redn, greenn, bluen,nirn))
    rgb = np.dstack((redn, greenn, bluen))
    outfile=writepath + '/' + image_folder + '.jpg'
    plt.imsave(outfile,rgb,format="jpg")

def get_data_stats(all_folders,categories,config,classids,writepath):
    select_type = config['selection_type']
    cnt_selected = 0
    data_selected=0
    cnt_summer = 0
    cnt_winter = 0
    cnt = 0
    all_record=[]
    select_record_summer=[]
    select_record_winter=[]
    selected_records=[]
    domainA = config['domainA']
    domainB = config['domainB']
    for folder in all_folders:
        file_with_path = folderpath + '/' + folder + '/' + folder + '_labels_metadata.json'
        with open(file_with_path) as f:
            data = json.load(f)
            label_array = list(set(config['label_array']).intersection(set( data['labels'])))
            class_array = []
            for label in label_array:
                class_array.append(classids[label])
            class_unique = list(set(class_array))    
            if len(class_unique) == 1:
                #print(label_array)
                label = label_array[0]
            #for label in data['labels']:
                filepart = file_with_path.split('_')[2]
                #if filepart[4:6] == '01' or filepart[4:6] == '12' or filepart[4:6] == '02':
                if filepart[4:6] in domainB:
                    record_type = 'winter'
                #elif filepart[4:6] == '06' or filepart[4:6] == '07' or filepart[4:6] == '08':
                elif filepart[4:6] in domainA: 
                    record_type = 'summer'
                else:
                    record_type = 'other'
                if record_type in ['winter' , 'summer']:
                    if select_type == 'all':
                        data_selected = 1
                    elif select_type == 'label' and label in config['label_array']:
                        data_selected = 1
                    elif select_type == 'category' and categories[label] == config['category']:
                        data_selected = 1
                    if data_selected == 1:
                        cnt_selected = cnt_selected + 1
                        imagepath = folderpath + '/' + folder + '/' + folder
                        visualizeGeoData(imagepath,writepath,folder)
                        if record_type == 'winter':
                            select_record_winter.append(folder)
                        else:
                            select_record_summer.append(folder)
                        imagefile = folder + '.jpg'    
                        classid = classids[label]
                        #print ('class:',classid)
                        selected_records.append([imagefile , folder,categories[label], label,classid])
                        data_selected=0
                            
                all_record.append([folder, categories[label], label, record_type])
        #print (all_record)
        cnt = cnt + 1
        #print (cnt)
        if (cnt  % 10000) == 0 :
            print (cnt)
    df=pd.DataFrame(all_record, columns=['folder' , 'category', 'label' , 'record_type'])
    new_df = df.groupby(['category' , 'label','record_type']).size().reset_index(name='counts') 
    selected_df = pd.DataFrame(selected_records, columns=['file' , 'folder' , 'category', 'label' , 'classid'])
    print ('selected:' , cnt_selected)
    select_record_winter = list(set(select_record_winter))
    select_record_summer = list(set(select_record_summer))
    return (df,new_df, cnt, cnt_selected,select_record_winter,select_record_summer,selected_df)

def get_splits(select_winter, select_summer,writepath,selected_df,domainA_folder):
    print (selected_df.classid.unique())
    cnt_winter = len(select_winter)
    cnt_summer = len(select_summer)
    cut_trainA = math.floor (cnt_summer/100)
    cut_trainB = math.floor (cnt_winter/100)
    cut_test = math.floor (2 * cnt_winter/3)
    model_train = math.floor ( 3 * cnt_summer/4)
    model_test = cnt_summer -(cut_trainA + model_train)
    model_validate = cnt_winter - (cut_trainB + cut_test)
    print ('Total Summer Images: ' , cnt_summer)
    print ('Total Winter Images: ' , cnt_winter)
    print ('CUT Train A:' , cut_trainA)
    print ('CUT Train B:' , cut_trainB)
    print ('CUT Predict B :' , cut_test)
    print ( 'Model Train :' , model_train)
    print ('Model Test :' , model_test)
    print ('Model Validate :' , model_validate)
    winter_rand = list(range(0, cnt_winter))
    summer_rand = list(range(0,cnt_summer))
    random.shuffle(winter_rand)
    random.shuffle(summer_rand)
    cut = writepath +'/CUT'
    cutdataset = cut + '/dataset'
    model = writepath + '/model'
    cuttrainA = writepath + '/CUT/dataset/trainA'
    cuttrainB = writepath + '/CUT/dataset/trainB'
    #cuttest = writepath + '/CUT/test'
    modeltrain = writepath + '/model/train'
    modeltest = writepath + '/model/test'
    modelvalidate = writepath + '/model/validate'
    cutpredict =  writepath +  '/CUT/predict'
    cutpredictdataset = cutpredict + '/dataset'
    cutprtrainA = cutpredictdataset + '/trainA'
    cutprtrainB = cutpredictdataset + '/trainB'
    #os.mkdir(writepath + '/CUT')
    #os.mkdir(writepath + '/model')
    os.mkdir(cut)
    os.mkdir(model)
    os.mkdir(cutdataset)
    os.mkdir(cuttrainA)
    os.mkdir(cuttrainB)
    #os.mkdir(cuttest)
    os.mkdir(modeltrain)
    os.mkdir(modeltest)
    os.mkdir(modelvalidate)
    os.mkdir(cutpredict)
    os.mkdir(cutpredictdataset)
    os.mkdir(cutprtrainA)
    os.mkdir(cutprtrainB)
    for classid in selected_df.classid.unique():
        classdir = writepath + '/model/train/' + classid
        os.mkdir(classdir)    
        classdir = writepath + '/model/test/' + classid
        os.mkdir(classdir)
        classdir = writepath + '/model/validate/' + classid
        os.mkdir(classdir)
    
    source_dir = writepath + '/alldata'
    cnt =  0
    #print (selected_df)
    if domainA_folder == '': 
        print ('ok')
    if domainA_folder == '':
        for i in summer_rand:
            source_file = source_dir + '/' + select_summer[i] + '.jpg'
            if cnt < cut_trainA:
                target = cuttrainA + '/' + select_summer[i] + '.jpg'
                shutil.copyfile(source_file , target)
            elif cnt < (cut_trainA + model_train):
                #print (select_summer[i])
                class_list = selected_df[selected_df['folder'] == select_summer[i]]['classid'].values.tolist() 
                for class_value in class_list:
                    target = modeltrain + '/' + class_value + '/' + select_summer[i] + '.jpg'
                    shutil.copyfile(source_file , target)
                target = cutprtrainA + '/' + select_summer[i] + '.jpg'
                shutil.copyfile(source_file , target)
            else:
                class_list = selected_df[selected_df['folder'] == select_summer[i]]['classid'].values.tolist()
                for class_value in class_list:
                    target = modeltest + '/'+ class_value + '/'  + select_summer[i] + '.jpg'
                    shutil.copyfile(source_file , target)
            if cnt == cut_trainA:
                print ("Completed CUT trainA generation")
            elif cnt == (cut_trainA + model_train):
                print ("Completed model train generation")
            if (cnt  % 2000) == 0 :
                print ('Summer Generation' , cnt)    
            cnt = cnt + 1    
    else:
        summer_source = domainA_folder + '/CUT/dataset/trainA'
        summer_target = writepath + '/CUT/dataset/trainA'
        shutil.rmtree(summer_target)
        shutil.copytree(summer_source,summer_target,symlinks=False, ignore=None)
        summer_source = domainA_folder + '/model/train/'
        summer_target = writepath + '/model/train'
        shutil.rmtree(summer_target)
        shutil.copytree(summer_source,summer_target,symlinks=False, ignore=None)
        summer_source = domainA_folder + '/CUT/predict/dataset/trainA'
        summer_target = writepath + '/CUT/predict/dataset/trainA'
        shutil.rmtree(summer_target)
        shutil.copytree(summer_source,summer_target,symlinks=False, ignore=None)
        summer_source = domainA_folder + '/model/test'
        summer_target = writepath + '/model/test'
        shutil.rmtree(summer_target)
        shutil.copytree(summer_source,summer_target,symlinks=False, ignore=None)
    print ("Completed model test generation")
    cnt = 0
    for i in winter_rand:
        source_file = source_dir + '/' + select_winter[i] + '.jpg'
        if cnt < cut_trainB:
            target = cuttrainB + '/' + select_winter[i] + '.jpg'
            shutil.copyfile(source_file , target)
     #       target = cutprtrainB + '/' + select_winter[i] + '.jpg'
     #       shutil.copyfile(source_file , target)
        elif cnt < (cut_trainB + cut_test):
            #target = cuttest + '/' + select_winter[i] + '.jpg'
            #shutil.copyfile(source_file , target)
            target = cutprtrainB + '/' + select_winter[i] + '.jpg'
            shutil.copyfile(source_file , target)
        else:
            class_list = selected_df[selected_df['folder'] == select_winter[i]]['classid'].values.tolist()
            for class_value in class_list:
                target = modelvalidate + '/'+ class_value + '/'  + select_winter[i] + '.jpg'
                shutil.copyfile(source_file , target)
        if cnt == cut_trainB:
            print ("Completed CUT trainB generation")
        elif cnt == (cut_trainB + cut_test):
            print ("Completed CUT Test generation")
        if (cnt  % 2000) == 0 :
            print ('Winter Data Generation' , cnt)     
        cnt = cnt + 1    
    print ("Completed model validate generation")
    print(cnt)
    print (cut_trainA,cut_trainB,cut_test,model_train,model_test)

### Main Section
start_time = time.time()
folderpath=sys.argv[1]
with open("dataselect_config.json") as f:
        config = json.load(f)
if config['class_type'] == 'category':
    categoryfile='category_label_all.json'
    classidfile = 'category_id_all.json'
else:    
    categoryfile='category_label.json'
    classidfile = 'category_id.json'
outdir=sys.argv[2]
outsummaryfile=outdir + '/' + 'labelsummary.csv'
outallfile=outdir + '/' + 'labelall.csv'
outselected = outdir + '/' + 'labelselected.csv'
flatselected = outdir + '/' +'flatselected.csv'
all_folders = listdir(folderpath)
print ('file_name' , ',' , 'label', ',' , 'record_type')
with open(classidfile) as f:
        classids = json.load(f)        
with open(categoryfile) as f:
        categories = json.load(f)
writepath= outdir + '/' + 'alldata'
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)
os.mkdir(writepath)
print ('Folder Created')
df,new_df,cnt , cnt_selected,select_winter , select_summer,selected_df = get_data_stats(all_folders,categories,config,classids,writepath)
print ('Image Name Selected')
if len(sys.argv) == 4:
    domainA_folder = sys.argv[3]
else:
    domainA_folder = ''
get_splits(select_winter, select_summer,outdir,selected_df,domainA_folder)
print (cnt , cnt_selected)
print(new_df)
new_df.to_csv(outsummaryfile, index=False)
df.to_csv(outallfile,index=False)
print(len(selected_df))
selected_df.to_csv(outselected,index=False)
selected_df_new = selected_df[['file' , 'classid']]
selected_df_new['classid'] = selected_df_new['classid'].astype(int)
#flat_df = selected_df_new.groupby(['file']).agg(lambda x: x.tolist())
flat_df = selected_df_new.groupby(['file']).agg(lambda x: list(dict.fromkeys(x.tolist())))
print(flat_df)
flat_df.to_csv(flatselected)
print("Exectuion Time :  %s seconds " % (time.time() - start_time))
