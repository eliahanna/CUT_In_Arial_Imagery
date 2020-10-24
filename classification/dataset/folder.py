# original source code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
import os
import os.path
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch.utils.data as data

from PIL import Image
import numpy as np
import pandas as pd
import os, time

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def tifffile_loader(path):
    # all the loader should be numpy ndarray [height, width, channels]
    # int16: (-32768 to 32767)
    import tifffile
    img = tifffile.imread(path)
    if img.dtype in [np.uint8, np.uint16, np.float]:
        return img
    else:
        raise TypeError('tiff file only support np.uint8, np.uint16, np.float, but got {}'.format(img.dtype))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # all the loader should be numpy ndarray [height, width, channels]
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img)


def image_loader(path):
    if os.path.splitext(path)[1].lower() in ['.tif', '.tiff']:
        return tifffile_loader(path)
    else:
        return pil_loader(path)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

#This method will be used where using multi-class classification , meaning one image is mapped to one class only
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


#This method will be used where using multi-class classification , meaning one image is mapped to one class only
class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        classes (callable, optional): List of the class names.
        class_to_idx (callable, optional): Dict with items (class_name, class_index).
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, classes=None, class_to_idx=None, transform=None, target_transform=None):
        if not class_to_idx:
            classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, **kwargs):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform, **kwargs)
        self.imgs = self.samples


# This will return a dataframe with the image full url and all the labels
# This method will be used for multi-label classification
def make_multi_dataset(dir, class_to_idx, extensions):
    script_start_time = time.time() # tells the total run time of this script
    columns=(sorted(class_to_idx.keys()))
    dictDataType={column:'Int64' for column in columns}
    columns.insert(0,'Image')
    columns.insert(1,'ImageName')
    df = pd.DataFrame(columns = columns )
    df= df.astype(dictDataType)
    dfdict = {}
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if(fname in dfdict) :
                        dfdict[fname][target]=1
                    else:
                        dfdict[fname]={columns[0]:path, columns[1] : fname , target : 1}
                    #if(df.Image.str.contains(fname).any()):
                     #   df.loc[df.Image.str.contains(fname),target] =1
                    #else:
                     #  df = df.append({columns[0]:path, columns[1] : fname , target : 1} , ignore_index=True)

    #tempDF = pd.DataFrame(columns = columns )
    #tempDF= tempDF.astype(dictDataType)
    df = pd.DataFrame.from_dict(dfdict, "index",columns = columns)
    df= df.astype(dictDataType)
    df=df.fillna(0)
    df.reset_index(inplace=True)
    df = df.drop(['index'], axis = 1)
    print("Original Dataframe ",df.iloc[:,1:])

    #print("dictionary Dataframe ",tempDF.iloc[:,1:])

    valCount= df.apply( lambda s : s.value_counts().get(key=1,default=0 ) , axis=0)
    sample_counts = valCount[2:].to_numpy()
    weight = 1. / (sample_counts)
    class_to_idx_keys=sorted(class_to_idx.keys())
    class_to_weights = {class_to_idx_keys[i]: weight[i] for i in range(len(class_to_idx_keys))}
    df['Weights'] = 0
    #print("Weights Counts ",weight)
    print ("class_to_weights: ",class_to_weights)
    for col in sorted(class_to_idx.keys()):
        df.loc[df[col] == 1, 'Weights'] = df['Weights'] + class_to_weights[col]


    sample_weights = df['Weights']
    #print(sample_weights)
    df = df.drop(['Weights'], axis = 1)
    #print(df.iloc[:,1:])
    time_elapsed = time.time() - script_start_time
    print("Time is dataloader: {} sec".format(round(time_elapsed)))
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    print('Time taken by the dataset to load is : {} h {} m !'.format(int(h), int(m)))
    return df,sample_weights.to_numpy()

# This dataloader is used for multi label classification
class ImageMultiLabelDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        One image can be under multiple class folders

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/xxy.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
        root/class_x/xxz.ext

        root/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        classes (callable, optional): List of the class names.
        class_to_idx (callable, optional): Dict with items (class_name, class_index).
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, classes=None, class_to_idx=None, transform=None, target_transform=None):
        if not class_to_idx:
            classes, class_to_idx = self._find_classes(root)

        print("classes ",classes)
        print("class_to_idx ",class_to_idx)

        samples,sample_weights = make_multi_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                           "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.sample_weights = sample_weights
        print("The shape of the dataset: ",samples.shape)
        #self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        row = self.samples.iloc[index, :]
        path=row.Image
        name=row.ImageName
        target = row[2:].values
        target = target.astype('double')
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target ,name

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import WeightedRandomSampler
    import torch
    traindir='/Users/adas1/Aditi/personal/school/210/bigearthnet/preprocess/output/model/train'

    #train_transform = {
    #    'none' : transforms.Compose([
    #        transforms.ToPILImage(),
    #        transforms.ToTensor(),
  #  #       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #    ]),
    #    'augment' : transforms.Compose([
    #        transforms.ToPILImage(),
    #        transforms.RandomHorizontalFlip(0.5),
     #       transforms.RandomVerticalFlip(0.5),
     #       transforms.RandomGrayscale(0.5),
     #       transforms.RandomRotation(40),
            #transforms.Grayscale(num_output_channels=3),
    #        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0),
     #       transforms.ToTensor(),
            #          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #    ])
    #}
    dataset_train = ImageMultiLabelDataset(root=traindir,transform=None)
    sample_weights = dataset_train.sample_weights
    print("sample Weoghts: ",sample_weights)

    samples_weights = torch.tensor(sample_weights)
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights),replacement=False)

    batch_size = 16

    dataloader = DataLoader(dataset_train, batch_size, num_workers=0,shuffle=False,sampler = sampler)
    #print("Length of the dataloader: ",len(dataloader))
    #fig, axs = plt.subplots(1, batch_size, figsize=(3,3))
    #print("length of the dataloader ",len(dataloader.dataset))
    #dataloader output 4 dimensional tensor - [batch, channel, height, width]. Matplotlib and other image processing libraries often requires [height, width, channel].
    for j, (image, target,name) in enumerate(dataloader):
        print("index ",j)
        print("Target ",target)
        if j==2:
            break

    #    for i in range(batch_size):
    #        print("i ",i ,image[i].shape)
    #        img =image[i] #/ 2 + 0.5
            #your transpose should convert a now [channel, height, width] tensor to a [height, width, channel] one. To do this, use np.transpose(image.numpy(), (1, 2, 0))
    #        img=np.transpose(img.numpy(), (1, 2, 0))
            #print("Values ", target[i].numpy(), image[i].numpy().shape)
    #        axs.imshow(img)

    #    break
    #plt.show()

