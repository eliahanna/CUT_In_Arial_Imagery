from os import listdir
import pandas as pd
import sys
import shutil
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
p = transforms.Compose([transforms.Resize((120,120))])
outdir=sys.argv[1]
inputfolder = sys.argv[2]
outselected = outdir + '/' + 'labelselected.csv'
selected_df  = pd.read_csv(outselected)
source_path = outdir + '/model/train'
target_path = outdir + '/model/CUTtrain/train'
if os.path.exists(outdir):
    shutil.rmtree(target_path)
shutil.copytree(source_path , target_path)
folderpath = outdir + '/model/CUTtrain/'
all_files = listdir(inputfolder)
for filename in all_files:
    if filename.endswith('.png'):
        folder = filename.split('.')[0]
        file_jpg = folder + '.jpg'
        file_fake = folder + '_fake.jpg'
        class_list = selected_df[selected_df['file'] == file_jpg]['classid'].values.tolist() 
        for class_value in class_list:
            source = inputfolder + filename
            im1 = Image.open(source)
            print (source)
            target = folderpath + 'train/' + str(class_value) + '/' + file_fake
            print (target)
            
            p(im1).save(target)
