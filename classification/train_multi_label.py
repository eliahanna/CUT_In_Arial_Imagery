import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from dataset.folder import ImageMultiLabelDataset
from model.utils import get_model


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
    correct = 0
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #TODO : We have to fix this evaluation matrix
            #correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(data_loader.dataset)/data_loader.batch_size

        #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    loss, correct, len(data_loader.dataset),
        #    100. * correct / len(data_loader.dataset)))
        print('\nTest set: Average loss: {:.4f}, Accuracy:  TBD)\n'.format(
            loss,
            ))
        writer.add_scalar('test/loss', loss, len(data_loader) * epoch)
        writer.add_scalar('test/accuracy', correct / len(data_loader.dataset), epoch)

#load the data as image and multiLabel
def load_data(traindir, valdir):
    #TODO: Find out the correct transformation
    train_transform = transforms.Compose([
 #       transforms.RandomHorizontalFlip(),
#        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = ImageMultiLabelDataset(root=traindir, transform=train_transform)
    dataset_val = ImageMultiLabelDataset(root=valdir, transform=val_transform)

    return dataset_train, dataset_val


def main(args):
    torch.backends.cudnn.benchmark = True

    #Step1. CPU or GPU
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
    # making empty lists to collect all the losses
    #TODO : populate this dictionay
    losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

    # Step2. load dataset and dataloader
    dataset_train, dataset_val = load_data(args.train_path, args.val_path)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

    print('\n-----Initial Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(dataset_train)))
    print('num images in val_dataset     : {}'.format(len(dataset_val)))
    print('-------------------------------------')

    a,b = dataset_train[0]
    print('\nwe are working with \nImages shape: {} and \nTarget shape: {}'.format( a.shape, b))

    # Step3. Instantiate the model
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    print("Model Summary: ", model.summary())
    model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

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


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Adaptation Classification Training')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--model', default="resnet18", help='the classification model')
    parser.add_argument('--pretrained', default=True, help='use the ImageNet pretrained model or not')

    parser.add_argument('--resume',default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')

    parser.add_argument('--device', default='cpu', help='the device platform for train, cuda or cpu.')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='training batch size')
    parser.add_argument('--epochs', default=90, type=int, help='train epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='checkpoint', help='path to save checkpoint')
   	# Additional arguments
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')	

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
