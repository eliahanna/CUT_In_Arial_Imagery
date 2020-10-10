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

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, writer):
    model.train()
    for idx, (image, target) in enumerate(data_loader):
        # Move tensors to the configured device
        image, target = image.to(device), target.to(device)
        # Clearing the last error gradient
        optimizer.zero_grad()
        # Forward pass
        output = model(image)
        # Calculate Loss
        loss = criterion(output, target)
        # Backward and optimize
        loss.backward()
        # Updating parameters
        optimizer.step()

        if idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(image), len(data_loader.dataset), 100. * idx / len(data_loader), loss.item()))
            writer.add_scalar('train/loss', loss.item(), len(data_loader) * epoch + idx)


def evaluate(epoch, model, criterion, data_loader, device, writer):
    model.eval()
    loss = 0
    accuracy_score = 0
    precision = 0
    recall = 0
    f1 =0
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss += criterion(output, target).item()

            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = (output > .5).int()

            accuracy_score = accuracy_score + hamming_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            precision = precision + precision_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            recall = recall + recall_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
            f1= f1 + f1_score(target.int().to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())

        #print("Number of batches : ",len(data_loader) , " and also : ",data_loader.batch_size)
        loss /= len(data_loader.dataset)/data_loader.batch_size

        #print(" Final Accuracy score : ",accuracy_score)
        accuracy_score /= len(data_loader)
        precision /= len(data_loader)
        recall /= len(data_loader)
        f1 /= len(data_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}, Precision: {} , Recall: {} , F1 score: {} \n'.format(
            loss, accuracy_score,precision,recall,f1))

        writer.add_scalar('test/loss', loss,  epoch)
        writer.add_scalar('test/accuracy', accuracy_score, epoch)
        writer.add_scalar('test/Precision', precision, epoch )
        writer.add_scalar('test/Recall', recall,epoch)
        writer.add_scalar('test/F1 score', f1 ,epoch)

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
            transforms.RandomGrayscale(0.2),
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

    #Step1. CPU or GPU
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
    script_start_time = time.time() # tells the total run time of this script
    #Test model
    if not args.test: # Training flow
        print("Training Flow ")


        # making empty lists to collect all the losses
        #TODO : populate this dictionay
        losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

        # Step2. load dataset and dataloader
        dataset_train, dataset_val = load_data(args.train_path, args.val_path,args.augmentation)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

        print('\n-----Initial Dataset Information-----')
        print('num images in train_dataset   : {}'.format(len(dataset_train)))
        print('num images in val_dataset     : {}'.format(len(dataset_val)))
        print('-------------------------------------')

        a,b = dataset_train[0]
        print('\nwe are working with \nImages shape: {} and \nTarget shape: {}'.format( a.shape, b))

        # Step3. Instantiate the model
        #model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
        #TODO: fix the model
        model = models.vgg16_bn(pretrained=True) # pretrained = False bydefault
        # change the last linear layer
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, args.num_classes)]) # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features) # Replace the model classifier

        #print(summary(model, input_size=(a.shape[0], a.shape[1], a.shape[2])))
        model.to(device)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))

        print('\nwe have {} Million trainable parameters here in the {} model'.format(count_parameters(model), model.__class__.__name__))

        # Step4. Binary Croos Entropy loss for multi-label classification

        criterion = nn.BCEWithLogitsLoss().to(device)

        # Step5. Adam optimizer and lr scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Let's not do the learning rate scheduler now
        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        writer = SummaryWriter(args.ckp_dir)
        for epoch in range(args.epochs):
            #writer.add_scalar('train/learning_rate', lr_scheduler.get_lr()[0], epoch)
            # Step6. Train the epoch
            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq, writer)
            #lr_scheduler.step()
            # Step7. Validate after each epoch
            evaluate(epoch, model, criterion, val_loader, device, writer)
            # Step8. Save the model after each epoch
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
        model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
        model.to(device)

        model.load_state_dict(torch.load(args.test_model, map_location=device))
        #Loss and optimizer
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        evaluate(1, model, criterion, test_loader, device, writer)
        writer.close()


    time_elapsed = time.time() - script_start_time
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print('{} h {}m is taken by the script to complete!'.format(int(h), int(m)))


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
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='checkpoint', help='path to save checkpoint')
   	# Additional arguments
    parser.add_argument('--drop', '--dropout', default=0, help='Dropout ratio')
    parser.add_argument('--test-path', help='test dataset path')
    parser.add_argument('--test-model',default='', help='path to latest checkpoint to run test on (default: none)')
    parser.add_argument('-t','--test', default=False, help = 'Set to true when running test')
    parser.add_argument('--augmentation', default=False, help = 'Set to true if you want to do data augmentation')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
