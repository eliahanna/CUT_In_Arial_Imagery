import argparse
import os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from dataset.folder import ImageMultiLabelDataset,ImageFolder
from model.utils import get_model
#from torchsummary import summary
import torchvision.models as models
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import collections
import logging
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import random


# define a function to count the total number of trainable parameters
def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, writer,logging,losses_dict,is_inception=False):
    model.train()
    epoch_loss = 0
    correct=0
    accuracy_score = 0
    for idx, (image, target,name) in enumerate(data_loader):
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
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred = np.squeeze(pred, axis = 1)
        correct += pred.eq(target.view_as(pred)).sum().item()
        #logging.info("Count of Correct Prediction {}".format(correct))
        
        if idx % print_freq == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(image), len(data_loader.dataset), 100. * idx / len(data_loader), loss.item()))
            writer.add_scalar('Train/loss', loss.item(), len(data_loader) * epoch + idx)

    epoch_loss /= len(data_loader.dataset)/data_loader.batch_size
    losses_dict['epoch_train_loss'].append(epoch_loss)
    accuracy_score = correct / len(data_loader.dataset)
    losses_dict['training_accuracy'].append(round(accuracy_score,4))
    logging.info('epoch_train_loss is {} and Train accuracy: {}'.format(epoch_loss, round(accuracy_score,4)))
    #logging.info(epoch_loss)

    #writer.add_scalar('Train/epoch loss', epoch_loss, epoch)

def evaluate(epoch, model, criterion, data_loader, device, writer,logging,losses_dict,validate=0):
    model.eval()
    loss = 0

    correct=0
    f1 =0
    classes= ('Arable Land','Pastures','Perm Crop','Heterogeneous Agricultural')
    #classes_dict= {0:'Arable Land',1:'Pastures',2:'Perm Crop',3:'Heterogenious Agricultural'}
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    checkDf = pd.DataFrame(columns = ['Image', 'Actual','Prediction','Matching'])
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for idx, (image, target,name) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            #logging.info("Output : {}".format(output))
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = np.squeeze(pred, axis = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()


            # Put everything in a dataframe for display
            match = pred.eq(target.view_as(pred))
            dicDf = {'Image': name,
             'Actual': target.int().to(torch.device("cpu")).numpy().tolist(),
             'Prediction': pred.to(torch.device("cpu")).numpy().tolist(),
             'Matching':match.to(torch.device("cpu")).numpy()
            }
            epochDf = pd.DataFrame(dicDf, columns = ['Image', 'Actual','Prediction','Matching'])
            checkDf = pd.concat([checkDf, epochDf], axis =0,ignore_index=True, sort=False)

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += match[i].item()
                class_total[label] += 1

        if epoch % 10 == 9:
            filePath=args.ckp_dir+'/validation_result_'+str(epoch)+'.csv'
            checkDf.to_csv(filePath)
        #print("Number of batches : ",len(data_loader) , " and also : ",data_loader.batch_size)
        loss /= len(data_loader)
        accuracy_score=correct / len(data_loader.dataset)

        logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
                    loss, round(accuracy_score,4)))
        if validate==1:
            losses_dict['epoch_val_loss'].append(loss)
            losses_dict['validation_accuracy'].append(round(accuracy_score,4))


        logging.info('\n-----Validation Accuracy by each label-----')
        for i in range(len(classes)):
            logging.info('Accuracy of {} : {} %'.format (
            classes[i], round(100 * class_correct[i] / class_total[i],2)))
        logging.info('\n---------------------------------------------')
        #writer.add_scalar('test/loss', loss,  epoch)
        #writer.add_scalar('test/accuracy', accuracy_score, epoch)
        #writer.add_scalar('test/Precision', precision, epoch )
        #writer.add_scalar('test/Recall', recall,epoch)
        #writer.add_scalar('test/F1 score', f1 ,epoch)


#load the data as image and multiLabel
def load_data(traindir, valdir):

    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #transforms.RandomGrayscale(0.2),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    inception_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        #transforms.RandomGrayscale(0.2),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.model == 'inception':
        dataset_train = ImageFolder(root=traindir, transform=inception_train_transform)
    else:
        dataset_train = ImageFolder(root=traindir, transform=train_transform)

    dataset_val = ImageFolder(root=valdir, transform=val_transform)

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
    #Train model
    if not args.test: # Training flow
        print("Training Flow ")

        # Step2. load dataset and dataloader
        dataset_train, dataset_val = load_data(args.train_path, args.val_path)
        sample_weights = dataset_train.sample_weights
        samples_weights = torch.tensor(sample_weights)
        weightedsampler = WeightedRandomSampler(samples_weights, len(samples_weights),replacement=True)

        #train_loader = DataLoader(dataset_train, batch_size=args.batch_size,  shuffle=False,sampler=weightedsampler)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,num_workers=0, shuffle=True,pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size,num_workers=0,shuffle=True,pin_memory=True)

        logging.info('\n-----Initial Dataset Information-----')
        logging.info('num images in train_dataset   : {}'.format(len(dataset_train)))
        logging.info('num images in val_dataset     : {}'.format(len(dataset_val)))
        logging.info('Size of train dataloader      : {}'.format(len(train_loader)))
        logging.info('Size of validation dataloader      : {}'.format(len(val_loader)))
        logging.info('-------------------------------------')

        a,b,c = dataset_train[0]
        print('\nwe are working with \n Image name: {} and \nImages shape: {} and \nTarget shape: {} '.format(c, a.shape, b))
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
        elif args.model == 'resnet18':
            #Resnet 18
            logging.info('Running model resnet18')
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            #model.fc = nn.Linear(num_ftrs, args.num_classes)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
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

        #print(summary(model, input_size=(a.shape[0], a.shape[1], a.shape[2])))
        model.to(device)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))

        print('\nwe have {} Million trainable parameters here in the {} model'.format(count_parameters(model), model.__class__.__name__))

        # Step4. Binary Croos Entropy loss for multi-label classification

        criterion = nn.CrossEntropyLoss().to(device)


        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

        # Step5. Adam optimizer and lr scheduler
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
        dataset_test = ImageFolder(root=args.test_path, transform=test_transform)
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
        elif args.model == 'resnet18':
            #Resnet 18
            logging.info('Running model resnet18')
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            #model.fc = nn.Linear(num_ftrs, args.num_classes)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
        elif args.model == 'resnet50':
            #Resnet 50
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            #model.fc = nn.Linear(num_ftrs, args.num_classes)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, args.num_classes)
            )
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
        criterion = nn.CrossEntropyLoss().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='checkpoint', help='path to save checkpoint')
   	# Additional arguments
    parser.add_argument('--test-path', help='test dataset path')
    parser.add_argument('--test-model',default='', help='path to latest checkpoint to run test on (default: none)')
    parser.add_argument('-t','--test', default=False, help = 'Set to true when running test')
    #parser.add_argument('--seed', default=0, help='seed value for deterministic model')

    args = parser.parse_args()
    return args

def set_seed(seed=1234):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    args = parse_args()
    main(args)
