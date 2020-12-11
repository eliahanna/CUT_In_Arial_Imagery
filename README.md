# CUT In Arial Imagery
## A CUT Approach for Domain Adaptation in Aerial Imagery

## References
* [Slide Show](https://docs.google.com/presentation/d/19oiKt0no4zbY6tm_8FUZn6wXmpXIvjLWI6HD_UWRF8M/edit?ts=5f60e106#slide=id.p)
* [Website](https://groups.ischool.berkeley.edu/DomainAdaptCUT/)
* [Presentation 2 Slide Show](https://docs.google.com/presentation/d/1BnX1egolFC08-aIfVxntyVcODXL1GEMiCQdiAgo2p0U/edit?usp=sharing)
* [Meeting Notes](https://docs.google.com/document/d/1yrmghJ7MDKia_e4599bPVZS--dsykM7bCIAFgfwfaDw/edit?usp=sharing)
* [Contrastive Learning for Unpaired Image-to-Image Translation](https://taesung.me/ContrastiveUnpairedTranslation/)
* [CUT github repo](https://github.com/taesungp/contrastive-unpaired-translation)
* [A Cycle GAN Approach for Heterogeneous Domain Adaptation in Land Use Classification](https://arxiv.org/abs/2004.11245v1)
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet)
* [Corine Land Cover Nomenclature Guidelines](https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html)
* [Cross-Domain Car Detection Using Unsupervised Image-to-Image Translation: From Day to Night](https://arxiv.org/pdf/1907.08719.pdf)
* [Team Project Plan and Agreement](https://docs.google.com/document/d/1R3yrOgeIXL21Eee40OmoX6BCMPBWow_t8RugZ02zvPo/edit?usp=sharing)


## System Access
* Access to IBM Cloud ```ssh -i id_rsa root@52.117.105.190```
* Access to AWS Instance ```ssh -i cut-aws.pem ubuntu@54.188.6.19```

## Preprocessing

1. The preprocessing creates the folders in a structure which can be consumed by domain transfer algorithms and calssification algorithms.
2. The program takes 3 file inputs . dataselect_config.json ( determines whether the program runs for all images , only particualar labels , only for a category) , category_id.json ( label mapping to ids. The ids would be used as classes for the classsification algorithm and folder generation , category_label.json ( mapping of labels to category).
3. Following are the folders that get generated : 
      a) output : This has all subfolders which are generated (model , cut). Also it has 3 files labelall.csv ( All labels for all images) , labelsummary.csv ( label , category counts) , labelselected.csv ( selected labels for the current run)
      a) Model : This has all the data relevant for classification. Model has 3 folders train , test , validate. Each folder has images stored inside by the label.
      b) CUT : This has the data relevant for the domain transfer model. It has 2 subfolders dataset , test. dataset folder has 2 subfolders trainA and trainB . It is used as an input for domain transfer model. The data in test folder is used for validating the efficacy of the domain transfer model.

4. The preprocessing step can be run as ```python3 data_classify.py /data/capstone/BigEarthNet-v1.0 output ``` . The first parameter is the path to the input image dataset , the second folder is the output directory for generated folders.
      
## Domain Transfer

1. For training the CUT model we need to run ```python3 train.py --dataroot /data/capstone/preprocess/output/CUT/dataset/ --name landform_CUT --CUT_mode CUT```. The dataroot folder is the location for CUT input data. The model output is strored in results folder.

2. Testing the CUT model ```python3 test.py --dataroot /data/capstone/preprocess/output/CUT/predict/dataset/ --name landform_CUT --CUT_mode CUT --phase train```.
      
Notes (add text later): cp results/landform_CUT/train_latest/images/fake_B/*.* /data/capstone/preprocess/output/model/CUTtrain  

python3 cut_to_model.py output

python3 train_cls.py --train-path /data/capstone/preprocess/output/model/CUTtrain/train/ --val-path /data/capstone/preprocess/output/model/test --model resnet18  --batch-size 16 --num-classes 3 --device cuda --ckp-dir /data/capstone/nn/results/results_fake/

python3 test_cls.py --val-path /data/capstone/preprocess/output/model/validate --ckp-dir /data/capstone/nn/results/results_fake/cls_epoch_89.pth --num-classes 3

python3 train_cls.py --train-path /data/capstone/preprocess/output/model/train/ --val-path /data/capstone/preprocess/output/model/test --model resnet18  --batch-size 16 --num-classes 3 --device cuda --ckp-dir /data/capstone/nn/results/results_wofake/



python3 test_cls.py --val-path /data/capstone/preprocess/output/model/validate --ckp-dir /data/capstone/nn/results/results_wofake/cls_epoch_89.pth --num-classes 3





