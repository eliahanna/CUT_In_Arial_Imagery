from PIL import Image
import argparse
import   os
import sys
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import models, transforms
import numpy as np
import pandas as pd
URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
#SUBDIR = '2750'
SUBDIR = 'train'
selected_url = '/data/capstone/preprocess/selected_flat.csv'
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # all the loader should be numpy ndarray [height, width, channels]
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img)

def default_loader_new(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)

# Apparently torchvision doesn't have any loader for this so I made one
# Advantage compared to without loader: get "for free" transforms, DataLoader
# (workers), etc
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        print('ok')
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        image = self.loader(self.paths[idx])
        path = self.paths[idx]
     #   print ('ok get' , self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1, path


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images




class CustomDataSet(Dataset):
    def __init__(self, root, loader=default_loader_new, transform=None):
        classes, class_to_idx = self._find_classes(root)
        self.root = root
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        samples = make_dataset(root, class_to_idx)
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.selected = pd.read_csv(selected_url) 
    
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        patharray = path.split('/')
        pathindex = len(patharray) - 1
        #sample = default_loader_new(path)
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        selected_data = self.selected[self.selected['file']==patharray[pathindex]]['classid']
        class_formatted = selected_data.iloc[0]
        #print (class_formatted)
        class_formatted = class_formatted.replace('[', '') 
        class_formatted = class_formatted.replace(']', '')
        class_formatted = class_formatted.replace(' ','')
        class_array = class_formatted.split(',')
        label_list = []
        #print(self.class_to_idx)
        onehot = np.zeros(len(self.classes))
        for class_name in class_array:
            if class_name != '13':
                onehot[self.class_to_idx[class_name]] = 1
                #label_list.append(self.class_to_idx[class_name])
        #print (label_list)    
        #print (onehot)
        return sample , target , onehot




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Predict the label on the specified files and outputs the results in csv format.
            If no file is specified, then run on the test set of EuroSAT and produce a report.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-m', '--model', default='weights/best.pt', type=str, help="Model to use for prediction"
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ##### Main Processing####
    
    save = torch.load(args.model, map_location='cpu')
    normalization = save['normalization']
    train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(**normalization)
  #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    tr = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(**normalization)])
    my_dataset = CustomDataSet('/data/capstone/preprocess/output/model/train',train_trans)
    train_loader = torch.utils.data.DataLoader(my_dataset , batch_size=32, shuffle=False,
                               num_workers=4, drop_last=True)
    for idx, img in enumerate(train_loader):
        print('')
    #    print(img[1])
