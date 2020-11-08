import pandas as pd
import sys
import shutil
import os
from os import listdir
from PIL import Image
import imagehash
from SSIM_PIL import compare_ssim
import time
start_time = time.time()
folder = sys.argv[1]
option = sys.argv[2]
folder_fake = folder + 'fake_B'
if option == 'A':
    folder_real = folder + 'real_A'
else:
    folder_real = folder + 'real_B'
all_real = listdir(folder_real)
cnt = 0
not_similar = 0
similar = 0
if option == 'A':
    outdir = folder + 'not_similar'
else:
    outdir = folder + 'similar'
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)
for real in all_real:
    real_source = folder_real + '/' + real
    fake_source = folder_fake + '/' + real
    im1 = Image.open(real_source)
    im2 = Image.open(fake_source)
    value = compare_ssim(im1, im2)
    cnt = cnt + 1
    target_path = outdir + '/' + real
    if option == 'A':
        if value < 0.8:
            shutil.copyfile(fake_source , target_path)
            not_similar = not_similar + 1
            print(real, value)
    else:
        if value > 0.3:
            shutil.copyfile(fake_source , target_path)
            similar = similar + 1
            print (real,value)
if option == 'A':
    print (cnt,not_similar)
else:
    print(cnt,similar)
end_time = time.time()
print ('Total time:' , end_time - start_time) 
