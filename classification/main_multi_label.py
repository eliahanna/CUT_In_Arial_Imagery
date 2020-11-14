import argparse
import os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from dataset.folder import ImageMultiLabelDataset
from model.utils import get_model
#from torchsummary import summary
import torchvision.models as models
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import collections
import logging
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet


# define a function to count the total number of trainable parameters
def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def recall_score(y_true, y_pred):
    rec_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )

        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/ \
                    float( len(set_true) )
        #print('tmp_a: {0}'.format(tmp_a))
        rec_list.append(tmp_a)
    return np.mean(rec_list)

def precision_score(y_true, y_pred):
    prec_list = []

    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        elif len(set_pred) == 0:
            tmp_a = 0
        else:
            tmp_a = len(set_true.intersection(set_pred))/ \
                    float( len(set_pred) )
        prec_list.append(tmp_a)
    return np.mean(prec_list)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )

        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/ \
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, writer,logging,losses_dict,is_inception=False):
    model.train()
    epoch_loss = 0
    accuracy_score = 0
    checkDf = pd.DataFrame(columns = ['Image', 'Actual','Prediction','Matching'])
    for idx, (image, target,name) in enumerate(data_loader):
        #logging.info("Target {}".format(target))
        # Move tensors to the configured device
        image, target = image.to(device), target.to(device)
        # Clearing the last error gradient
        optimizer.zero_grad()

        if is_inception:
            output, aux_output = model(image)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            # Forward pass
            output = model(image)
            # Calculate Loss
            loss = criterion(output, target)

        # Backward and optimize
        loss.backward()
        # Updating parameters
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()
        pred_opt = torch.sigmoid(output)
        #logging.info("Prediction {}".format(pred_opt))
        pred = (pred_opt > .5).int()

        # Put everything in a dataframe for display
        match = np.all(target.int().to(torch.device("cpu")).numpy()==pred.to(torch.device("cpu")).numpy() , axis=1)
        dicDf = {'Image': name,
                 'Actual': target.int().to(torch.device("cpu")).detach().numpy().tolist(),
                 'Prediction': pred_opt.to(torch.device("cpu")).detach().numpy().tolist(),
                 'Matching':match
                 }
        epochDf = pd.DataFrame(dicDf, columns = ['Image', 'Actual','Prediction','Matching'])
        checkDf = pd.concat([checkDf, epochDf], axis =0,ignore_index=True, sort=False)

        accuracy_score = accuracy_score + hamming_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
        if idx % print_freq == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(image), len(data_loader.dataset), 100. * idx / len(data_loader), loss.item()))
            writer.add_scalar('Train/loss', loss.item(), len(data_loader) * epoch + idx)

    if epoch % 10 == 9 :
        filePath=args.ckp_dir+'/training_result_'+str(epoch)+'.csv'
        checkDf.to_csv(filePath)

    epoch_loss /= len(data_loader.dataset)/data_loader.batch_size
    losses_dict['epoch_train_loss'].append(epoch_loss)
    accuracy_score /= len(data_loader)
    losses_dict['training_accuracy'].append(round(accuracy_score,4))
    logging.info('Epoch train loss is {} and Train accuracy: {}'.format(epoch_loss, round(accuracy_score,4)))
    #logging.info(epoch_loss)

    #writer.add_scalar('Train/epoch loss', epoch_loss, epoch)

def evaluate(epoch, model, criterion, data_loader, device, writer,logging,losses_dict,validate=0):
    model.eval()
    loss = 0
    accuracy_score = 0
    precision = 0
    recall = 0
    f1 =0
    checkDf = pd.DataFrame(columns = ['Image', 'Actual','Prediction','Matching'])
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for idx, (image, target,name) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            pred_opt = torch.sigmoid(output)
            loss += criterion(output, target).item()

            pred = (pred_opt > .5).int()

            # Put everything in a dataframe for display
            match = np.all(target.int().to(torch.device("cpu")).numpy()==pred.to(torch.device("cpu")).numpy() , axis=1)
            dicDf = {'Image': name,
             'Actual': target.int().to(torch.device("cpu")).numpy().tolist(),
             'Prediction': pred_opt.to(torch.device("cpu")).numpy().tolist(),
             'Matching':match
            }
            epochDf = pd.DataFrame(dicDf, columns = ['Image', 'Actual','Prediction','Matching'])
            checkDf = pd.concat([checkDf, epochDf], axis =0,ignore_index=True, sort=False)

            accuracy_score = accuracy_score + hamming_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            precision = precision + precision_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            recall = recall + recall_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            f1= f1 + f1_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())

        if epoch % 10 == 9 or epoch % 10 == 5:
            filePath=args.ckp_dir+'/validation_result_'+str(epoch)+'.csv'
            checkDf.to_csv(filePath)
        #print("Number of batches : ",len(data_loader) , " and also : ",data_loader.batch_size)
        loss /= len(data_loader)

        print(" Total Accuracy score : ",accuracy_score)
        accuracy_score /= len(data_loader)
        precision /= len(data_loader)
        recall /= len(data_loader)
        f1 /= len(data_loader)
        logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}, Precision: {} , Recall: {} , F1 score: {} \n'.format(
            loss, round(accuracy_score,4),round(precision,4),round(recall,4),round(f1,4)))

        if validate==1:
            losses_dict['epoch_val_loss'].append(loss)
            losses_dict['validation_accuracy'].append(round(accuracy_score,4))
            losses_dict['validation_F1_score'].append((round(f1,4)))

        #writer.add_scalar('test/loss', loss,  epoch)
        #writer.add_scalar('test/accuracy', accuracy_score, epoch)
        #writer.add_scalar('test/Precision', precision, epoch )
        #writer.add_scalar('test/Recall', recall,epoch)
        #writer.add_scalar('test/F1 score', f1 ,epoch)


#load the data as image and multiLabel
def load_data(traindir, valdir,augmentation=False):
    #TODO: Find out the correct transformation
    transformation='none'
    train_transform = {
        'none' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'augment' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #transforms.RandomGrayscale(0.2),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if augmentation:
        transformation='augment'

    print("Train time transformation : ",transformation)
    dataset_train = ImageMultiLabelDataset(root=traindir, transform=train_transform[transformation])
    dataset_val = ImageMultiLabelDataset(root=valdir, transform=val_transform)

    return dataset_train, dataset_val


def main(args):
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.ckp_dir + '/' +args.model+".log"),
            logging.StreamHandler()
        ]
    )


    #Step1. CPU or GPU
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
    script_start_time = time.time() # tells the total run time of this script
    # making empty lists to collect all the losses
    losses_dict = collections.defaultdict(list)
    #losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}
    #Train model
    if not args.test: # Training flow
        print("Training Flow ")

        # Step2. load dataset and dataloader
        dataset_train, dataset_val = load_data(args.train_path, args.val_path,args.augmentation)
        sample_weights = dataset_train.sample_weights
        samples_weights = torch.tensor(sample_weights)
        weightedsampler = WeightedRandomSampler(samples_weights, len(samples_weights),replacement=True)

        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,sampler=weightedsampler)
        #train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

        logging.info('\n-----Initial Dataset Information-----')
        logging.info('num images in train_dataset   : {}'.format(len(dataset_train)))
        logging.info('num images in val_dataset     : {}'.format(len(dataset_val)))
        logging.info('Size of train dataloader      : {}'.format(len(train_loader)))
        logging.info('Size of validation dataloader      : {}'.format(len(val_loader)))
        logging.info('-------------------------------------')

        a,b,c = dataset_train[0]
        print('\nwe are working with \n Image name: {} and \nImages shape: {} and \nTarget shape: {} '.format(c, a.shape, b))
#        label_weight=torch.as_tensor(label_weight, dtype=torch.float).to(device)
        #dataiter = iter(train_loader)
        #images, labels,name = dataiter.next()
        #print("sampe :",images.shape)
        is_inception=False
        # Step3. Instantiate the model
        logging.info('Running model {} '.format(args.model))

        if args.model == 'vgg16':
            #VGG16
            logging.info('Running model vgg16')
            model = models.vgg16_bn(pretrained=True) # pretrained = False bydefault
            # change the last linear layer
            num_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(num_features, args.num_classes)]) # Add our layer with 4 outputs
            model.classifier = nn.Sequential(*features) # Replace the model classifier
        elif args.model == 'resnet50':
            #Resnet 50
            logging.info('Running model resnet50')
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            #model.fc = nn.Linear(num_ftrs, args.num_classes)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
        elif args.model == 'resnet101':
            logging.info('Running model resnet101')
            #Resnet 101
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.num_classes)
        elif args.model == 'efficientnet':
            logging.info('Running model efficientnet')
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.num_classes)
        elif args.model == 'inception':
            logging.info('Running model inception')
            model = models.inception_v3(pretrained=True)
            # Handle the auxilary net
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs_aux, args.num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,args.num_classes)
            is_inception=True

        logging.info(model)

        #logging.info(summary(model, input_size=(3, 299, 299)))
        model.to(device)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))

        print('\nwe have {} Million trainable parameters here in the {} model'.format(count_parameters(model), model.__class__.__name__))

        # Step4. Binary Croos Entropy loss for multi-label classification

        #criterion = nn.BCEWithLogitsLoss(pos_weight=label_weight).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)


        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

        # Step5. Adam optimizer and lr scheduler
        #optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
        #optimizer = optim.RMSprop(model.parameters(), lr = args.lr, alpha = 0.9)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # Let's not do the learning rate scheduler now
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        writer = SummaryWriter(args.ckp_dir)
        for epoch in range(args.epochs):
            # Step6. Train the epoch
            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq, writer,logging,losses_dict,is_inception)
            #lr_scheduler.step()
            # Step7. Validate after each epoch
            evaluate(epoch, model, criterion, val_loader, device, writer,logging,losses_dict,1)
            # Step8. Save the model after 10 epoch
            if epoch % 10 == 9:
                torch.save(model.state_dict(), os.path.join(args.ckp_dir, "cls_epoch_{}.pth".format(epoch)))
        writer.close()
    else:
        print("Testing Flow")
        writer = SummaryWriter(args.ckp_dir)
        #Load data
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_test = ImageMultiLabelDataset(root=args.test_path, transform=test_transform)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
        #Load model
        #model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
        if args.model == 'vgg16':
            #VGG16
            model = models.vgg16_bn(pretrained=True) # pretrained = False bydefault
            # change the last linear layer
            num_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(num_features, args.num_classes)]) # Add our layer with 4 outputs
            model.classifier = nn.Sequential(*features) # Replace the model classifier
        elif args.model == 'resnet50':
            #Resnet 50
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.num_classes)
            #model.fc = nn.Sequential(
            #    nn.Dropout(0.5),
            #    nn.Linear(num_ftrs, args.num_classes)
            #)
        elif args.model == 'resnet101':
            #Resnet 50
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.num_classes)
        elif args.model == 'efficientnet':
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.num_classes)
        elif args.model == 'inception':
            model = models.inception_v3(pretrained=True)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,args.num_classes)

        model.to(device)

        model.load_state_dict(torch.load(args.test_model, map_location=device))
        #Loss and optimizer
        criterion = nn.BCEWithLogitsLoss().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
        #optimizer = optim.RMSprop(model.parameters(), lr = args.lr, alpha = 0.9)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        evaluate(1, model, criterion, test_loader, device, writer,logging,losses_dict)
        writer.close()


    time_elapsed = time.time() - script_start_time
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    logging.info('{} h {}m is taken by the script to complete!'.format(int(h), int(m)))
    logging.info('losses At the end {}'.format(losses_dict))

    #plot and save graph
    if not args.test: # Training flowf
        plt.plot(losses_dict['epoch_train_loss'])
        plt.plot(losses_dict['epoch_val_loss'])
        plt.title('Model Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(args.ckp_dir+'/losses.png', bbox_inches='tight')
        plt.close()

        plt.plot(losses_dict['training_accuracy'])
        plt.plot(losses_dict['validation_accuracy'])
        plt.title('Model Evaluation')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(args.ckp_dir+'/accuracy.png', bbox_inches='tight')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Domain Adaptation Classification Training')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--model', default="vgg16", help='the classification model')
    parser.add_argument('--pretrained', default=True, help='use the ImageNet pretrained model or not')

    parser.add_argument('--resume',default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')

    parser.add_argument('--device', default='cpu', help='the device platform for train, cuda or cpu.')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='training batch size')
    parser.add_argument('--epochs', default=90, type=int, help='train epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='checkpoint', help='path to save checkpoint')
   	# Additional arguments
    parser.add_argument('--test-path', help='test dataset path')
    parser.add_argument('--test-model',default='', help='path to latest checkpoint to run test on (default: none)')
    parser.add_argument('-t','--test', default=False, help = 'Set to true when running test')
    parser.add_argument('--augmentation', default=False, help = 'Set to true if you want to do data augmentation')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
