# CUT In Arial Imagery
## A CUT Approach for Domain Adaptation in Aerial Imagery

## Classification
This directory or folder will consist of code related to classification model (for e.g. VGG,ResNET)

## Multi-Label classification 
This model is currently trained with BigEarthNet dataset that has multiple labels per image. Hence, our classification model currently supports multi-label classification. The default model is VGG16.

## How to run the Classification Model
The main python script main_multi_label.py which takes in the following parameter:
  --train-path TRAIN_PATH train dataset path
  --val-path VAL_PATH]  validation dataset path
  --model The classification model you are using, default to VGG16
  --pretrained PRETRAINED If you want use the ImageNet pretrained model or not, default=True
  --num-classes NUM_CLASSES Number of labels you are predicting
  --resume resume Path to the latest checkpoint (default: none)
  --in-channels IN_CHANNELS Input image channel
  --device DEVICE the device platform for train, cuda or cpu. default = cpu
  -b BATCH_SIZE training batch size , default=16
  --epochs EPOCHS How many epoch to train , default=90
  --lr LR initial learning rate , default=0.1
  --print-freq PRINT_FREQ] 
  --drop Dropout] 
  --test-path TEST_PATH
  --test-model path to latest checkpoint to run test on (default: none)
  -t TEST Whether you are running test or not , default=False

### Example to run Training

### Example to run Test
