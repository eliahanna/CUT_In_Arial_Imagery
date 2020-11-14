from os import listdir
import json
import sys
import pandas as pd
import time
import rasterio
import numpy as np
import shutil
import os
outputpath=sys.argv[2]
sourcepath = sys.argv[1]
#summer_array = ['06','07','08']
#winter_array = ['12' , '01' , '02']
with open("dataselect_config.json") as f:
        config = json.load(f)

summer_array = config['domainA']
winter_array = config['domainB']

if config['class_type'] == 'category':
    categoryfile='category_label_all.json'
    classidfile = 'category_id_all.json'
    labels= config['label_array']
print (labels)    
with open(classidfile) as f:
        classids = json.load(f)
with open(categoryfile) as f:
        categories = json.load(f)

## Count Total Images ##
cnt_summer = 0
cnt_winter = 0
all_folders = listdir(sourcepath)
for folder in all_folders:
    file_with_path = sourcepath + '/' + folder + '/' + folder + '_labels_metadata.json'
    with open(file_with_path) as f:
            data = json.load(f)
            flg_summer = 0
            flg_winter = 0
            for label in data['labels']:
                if label in labels:
                    filepart = file_with_path.split('_')[2]
                    #if filepart[4:6] == '06' or filepart[4:6] == '07' or filepart[4:6] == '08' :
                    if filepart[4:6] in summer_array:
                        flg_summer = 1
                    elif filepart[4:6] in winter_array:
                    #elif filepart[4:6] == '12' or filepart[4:6] == '01' or filepart[4:6] == '02' :
                        flg_winter = 1
            if flg_summer == 1:
                cnt_summer = cnt_summer + 1
                flg_summer = 0
            if flg_winter == 1:
                cnt_winter = cnt_winter + 1
                flg_winter = 1
cnt_total = cnt_summer + cnt_winter 
print ('****All Data Validation****')
print ('Total Images to be generated: ', cnt_total)
alldata = listdir(outputpath + '/alldata')
cnt_all = 0
for folder in alldata:
    cnt_all = cnt_all + 1
print ('Total Images generated: ' , cnt_all)    
if cnt_total == cnt_all:
    print('Total Number of Images Correct')
else:
    print('Total Number  of Images Incorrect')

print ('****Validation of Summer and Winter Counts****')
print ('Total summer images to be generated:' , cnt_summer)
print ('Total winter images to be generated:' , cnt_winter)
modeltrain_folder = outputpath + '/model/train'
all_folders = listdir(modeltrain_folder)
train = []
domain = []
class_metadata = []
class_generated = []
for subfolder in all_folders:
    subfoldername = outputpath + '/model/train/' + subfolder
    subfolders = listdir(subfoldername)
    for folder in subfolders:
        train.append(folder)
        domain.append(folder.split('_')[2][4:6])
## start label validate ##
        imagename = folder.replace('.jpg', '')
        imagemetadata = sourcepath + '/' + imagename + '/' + imagename + '_labels_metadata.json'
        with open(imagemetadata) as f:
            metadata = json.load(f)
            classes = []
            for label in metadata['labels']:
                classes.append(classids[label])
        class_generated.append([folder,subfolder])        
        class_metadata.append([folder,classes])
df_class_metadata = pd.DataFrame(class_metadata, columns=['image' , 'classes'])
df_class_metadata = df_class_metadata.drop_duplicates(subset='image', keep="first")
df_class_generated=pd.DataFrame(class_generated, columns=['image' , 'classes'])   
df_class_generated_flat = df_class_generated.groupby(['image'] , as_index=False ).agg(lambda x: x.tolist())
df_class_metadata.set_index('image',inplace=True)
df_class_generated_flat.set_index('image',inplace=True)
intersect_df_1=df_class_metadata.drop(df_class_generated_flat.index)
intersect_df_2=df_class_generated_flat.drop(df_class_metadata.index)
if intersect_df_1.empty and intersect_df_2.empty:
    label_val_text = 'Generated and source class labels match'
else:
    label_val_text = 'Generated and source class labels do not match'
## End of label validate

train = list(set(train))   
print ('Images in model train:' , len(train), ', Domain:', list(set(domain)) , ' , ', label_val_text)

modeltest_folder = outputpath + '/model/test'
all_folders = listdir(modeltest_folder)
test = []
domain = []
class_metadata = []
class_generated = []
for subfolder in all_folders:
    subfoldername = outputpath + '/model/test/' + subfolder
    subfolders = listdir(subfoldername)
    for folder in subfolders:
        test.append(folder)
        domain.append(folder.split('_')[2][4:6])
## start label validate ##
        imagename = folder.replace('.jpg', '')
        imagemetadata = sourcepath + '/' + imagename + '/' + imagename + '_labels_metadata.json'
        with open(imagemetadata) as f:
            metadata = json.load(f)
            classes = []
            for label in metadata['labels']:
                classes.append(classids[label])
        class_generated.append([folder,subfolder])
        class_metadata.append([folder,classes])
df_class_metadata = pd.DataFrame(class_metadata, columns=['image' , 'classes'])
df_class_metadata = df_class_metadata.drop_duplicates(subset='image', keep="first")
df_class_generated=pd.DataFrame(class_generated, columns=['image' , 'classes'])
df_class_generated_flat = df_class_generated.groupby(['image'] , as_index=False ).agg(lambda x: x.tolist())
df_class_metadata.set_index('image',inplace=True)
df_class_generated_flat.set_index('image',inplace=True)
intersect_df_1=df_class_metadata.drop(df_class_generated_flat.index)
intersect_df_2=df_class_generated_flat.drop(df_class_metadata.index)
if intersect_df_1.empty and intersect_df_2.empty:
    label_val_text = 'Generated and source class labels match'
else:
    label_val_text = 'Generated and source class labels do not match'
## End of label validate

test = list(set(test))                 
print('Images in model test:' , len(test) ,', Domain:', list(set(domain)) , ' , ', label_val_text)

modelvalidate_folder = outputpath + '/model/validate'
all_folders = listdir(modelvalidate_folder)
validate = []
domain = []
class_metadata = []
class_generated = []
for subfolder in all_folders:
    subfoldername = outputpath + '/model/validate/' + subfolder
    subfolders = listdir(subfoldername)
    for folder in subfolders:
        validate.append(folder)
        domain.append(folder.split('_')[2][4:6])
## start label validate ##
        imagename = folder.replace('.jpg', '')
        imagemetadata = sourcepath + '/' + imagename + '/' + imagename + '_labels_metadata.json'
        with open(imagemetadata) as f:
            metadata = json.load(f)
            classes = []
            for label in metadata['labels']:
                classes.append(classids[label])
        class_generated.append([folder,subfolder])
        class_metadata.append([folder,classes])
df_class_metadata = pd.DataFrame(class_metadata, columns=['image' , 'classes'])
df_class_metadata = df_class_metadata.drop_duplicates(subset='image', keep="first")
df_class_generated=pd.DataFrame(class_generated, columns=['image' , 'classes'])
df_class_generated_flat = df_class_generated.groupby(['image'] , as_index=False ).agg(lambda x: x.tolist())
df_class_metadata.set_index('image',inplace=True)
df_class_generated_flat.set_index('image',inplace=True)
intersect_df_1=df_class_metadata.drop(df_class_generated_flat.index)
intersect_df_2=df_class_generated_flat.drop(df_class_metadata.index)
if intersect_df_1.empty and intersect_df_2.empty:
    label_val_text = 'Generated and source class labels match'
else:
    label_val_text = 'Generated and source class labels do not match'
## End of label validate

validate = list(set(validate))
print('Images in model validate:' , len(validate) , ', Domain:', list(set(domain)) , ' , ', label_val_text)

cuttrainA_folder = outputpath + '/CUT/dataset/trainA'
all_folders = listdir(cuttrainA_folder)
cuttrainA = []
domain = []
for folder in all_folders:
    domain.append(folder.split('_')[2][4:6])
    cuttrainA.append(folder)
print('Images in CUT trainA:' , len(cuttrainA) , ', Domain:', list(set(domain)))  

cuttrainB_folder = outputpath + '/CUT/dataset/trainB'
all_folders = listdir(cuttrainB_folder)
cuttrainB = []
domain = []
for folder in all_folders:
    cuttrainB.append(folder)
    domain.append(folder.split('_')[2][4:6])
print('Images in CUT trainB:' , len(cuttrainB) , ', Domain:', list(set(domain)))

cuttestA_folder = outputpath + '/CUT/predict/dataset/trainA'
all_folders = listdir(cuttestA_folder)
cuttestA = []
domain = []
for folder in all_folders:
    cuttestA.append(folder)
    domain.append(folder.split('_')[2][4:6])
print('Images in CUT predict trainA:' , len(cuttestA) , ', Domain:', list(set(domain)))

cuttestB_folder = outputpath + '/CUT/predict/dataset/trainB'
all_folders = listdir(cuttestB_folder)
cuttestB = []
domain = []
for folder in all_folders:
    cuttestB.append(folder)
    domain.append(folder.split('_')[2][4:6])
print('Images in CUT predict trainB:' , len(cuttestB) , ', Domain:', list(set(domain)))

print('****Validation of overlaps***')
intersect = 0
intersect = intersect + len(set(train).intersection(set(test))) 
intersect = intersect + len(set(train).intersection(set(validate)))
intersect = intersect + len(set(train).intersection(set(cuttrainA)))
intersect = intersect + len(set(train).intersection(set(cuttrainB)))
intersect = intersect + len(set(train).intersection(set(cuttestB)))
intersect = intersect + len(set(test).intersection(set(validate)))
intersect = intersect + len(set(test).intersection(set(cuttrainA)))
intersect = intersect + len(set(test).intersection(set(cuttrainB)))
intersect = intersect + len(set(test).intersection(set(cuttestB)))
intersect = intersect + len(set(validate).intersection(set(cuttrainA)))
intersect = intersect + len(set(validate).intersection(set(cuttrainB)))
intersect = intersect + len(set(validate).intersection(set(cuttestB)))
intersect = intersect + len(set(cuttrainA).intersection(set(cuttrainB)))
intersect = intersect + len(set(cuttrainA).intersection(set(cuttestB)))
intersect = intersect + len(set(cuttrainB).intersection(set(cuttestB)))
print ('Overlap:' , intersect)

intersect = len(set(train).intersection(set(cuttestA)))
if intersect == len(train):
    print ('100% overlap of train and CUT test A')
else:
    print('train and CUT test A are not similar')
