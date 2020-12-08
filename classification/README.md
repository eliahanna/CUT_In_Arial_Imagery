    
# Classification modeling for Domain Adaptation in Aerial Imagery
This directory consists of code related to classification model (for e.g. VGG,ResNET,EfficientNet). It contains code for both multi-class classification and multi-label classification.
Both of them support VGG-16, Resnet-50, Resnet-101 and EfficientNet.

- Use **main_multi_class.py** for multi-class classification.
- Use **main_multi_label.py** for multi-label classification.

## Data 
Our models are currently trained with BigEarthNet dataset that has multiple labels per image. Hence, our classification model currently supports multi-label classification. 
You may also train with single images per class. The default model is VGG16. Augmentation is default in multi-class whereas it is optional for multi-label.

## How to run the Classification Model
The main python script is main_multi_label.py or main_multi_class.py which takes in the following parameter:

  *--train-path TRAIN_PATH -> training dataset path
  
  *--val-path VAL_PATH  ->validation dataset path
  
  *--model -> The classification model you are using, default is VGG16
  
  *--num-classes NUM_CLASSES -> Number of labels you are predicting
  
  *--resume resume -> Path to the latest checkpoint during training (default: none)
  
  *--device DEVICE -> the device platform for train, cuda or cpu. default = cpu
  
  *--b BATCH_SIZE -> training batch size , default=16
  
  *--epochs EPOCHS -> How many epoch to train , default=90
  
  *--lr LR -> initial learning rate , default=0.001
  
  *--print-freq PRINT_FREQ
  
  *--test-path TEST_PATH -> prediction/test dataset path
  
  *--test-model -> path to latest checkpoint model to run test on (default: none)
  
  *--t TEST -> Whether you are running test or not , default=False
  
  *--augmentation -> the default is false, turn it to true if you want to use data augmentation

### Example to run Training for multi-label
python3 main_multi_label.py --train-path ../preprocess/output/model/train \
                 --val-path ../preprocess/output/model/validate \
                 --model vgg16 \
                 --batch-size 32 \
                 --num-classes 11 \
                 --ckp-dir result	\
                 --epochs 2	\
                 --device cuda \
                 --augmentation True
                 

### Example to run Test for multi-label
python3 main_multi_label.py --test True --test-path ../preprocess/output/model/test \
                 --test-model result/cls_epoch_1.pth \
                 --model vgg16 \
                 --batch-size 32 \
                 --num-classes 11 \
                 --ckp-dir result  \
                 --epochs 2 \
                 --device cuda

### Example to run Training for multi-class
python3 main_multi_class.py --train-path ../preprocess/outputsf/model/CUT400/train \
                 --val-path ../preprocess/outputsf/model/test \
                 --model vgg16 \
                 --batch-size 256 \
                 --num-classes 4 \
                 --ckp-dir vgg16_sf_fake_400 \
                 --epochs 50 --lr 0.0001  \
                 --device cuda	 --print-freq 100

### Example to run Training for multi-class
python3 main_multi_class.py --test True --test-path ../preprocess/outputsf/model/validate \
                 --test-model vgg16_sf_fake_400/cls_epoch_49.pth \
                 --model vgg16 \
                 --batch-size 256 \
                 --num-classes 4 \
                 --ckp-dir vgg16_sf_fake_400  \
                 --device cuda
