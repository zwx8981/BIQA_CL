import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from itertools import combinations


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


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


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    if I.mode != 'RGB':
        I = I.convert('RGB')
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 verbose=False,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader,
                 task_id=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        if verbose:
            print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None)

        if task_id is not None:
            if task_id == 0:
                self.data = self.data.iloc[0:7000]
            elif task_id == 1:
                self.data = self.data.iloc[0:15000]
            elif task_id == 2:
                self.data = self.data.iloc[0:25000]
            elif task_id == 3:
                self.data = self.data.iloc[0:45000]
            elif task_id == 4:
                self.data = self.data.iloc[0:140000]


        if verbose:
            print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)

            mos = self.data.iloc[index, 1]
            std = self.data.iloc[index, 2]
            sample = {'I': I, 'mos': mos, 'std': std}
        else:
            image_name1 = os.path.join(self.img_dir, self.data.iloc[index, 0])
            image_name2 = os.path.join(self.img_dir, self.data.iloc[index, 1])

            I1 = self.loader(image_name1)
            I2 = self.loader(image_name2)
            y = torch.FloatTensor(self.data.iloc[index, 2:].tolist())
            if self.transform is not None:
                I1 = self.transform(I1)
                I2 = self.transform(I2)

            #y[0] = torch.clamp(y[0], 0.01,0.99)

            sample = {'I1': I1, 'I2': I2, 'y': y[0], 'std1': y[1], 'std2': y[2], 'yb': y[3]}

        return sample

    def __len__(self):
        return len(self.data.index)


class ImageDataset_pandas(Dataset):
    def __init__(self, pandas_object,
                 img_dir,
                 num_sample=1000,
                 transform=None,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """

        print('start loading csv data...')
        self.data = pandas_object
        self.num_data = len(self.data)
        all_index = range(0, self.num_data)
        all_combinations = list(combinations(all_index, 2))
        all_combinations = np.array(all_combinations)
        num_pairs = len(all_combinations)

        pair_index = np.arange(0, num_pairs)

        if num_pairs > num_sample:
            np.random.shuffle(pair_index)
            pair_index = pair_index[0:num_sample]

        self.all_pairs = all_combinations[pair_index]
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        pair = self.all_pairs[index]
        image_name1 = os.path.join(self.img_dir, self.data.iloc[pair[0], 0])
        image_name2 = os.path.join(self.img_dir, self.data.iloc[pair[1], 0])

        I1 = self.loader(image_name1)
        I2 = self.loader(image_name2)
        #y = torch.FloatTensor(self.data.iloc[index, 2:].tolist())

        if self.data.iloc[pair[0], 1] > self.data.iloc[pair[1], 1]:
            yb = 1
        else:
            yb = 0

        yb = torch.Tensor([yb])

        if self.transform is not None:
            I1 = self.transform(I1)
            I2 = self.transform(I2)

        # y[0] = torch.clamp(y[0], 0.01,0.99)


        #sample = {'I1': I1, 'I2': I2, 'y': y[0], 'std1': y[1], 'std2': y[2], 'yb': y[3]}
        sample = {'I1': I1, 'I2': I2, 'yb':yb}
        return sample

    def __len__(self):
        return len(self.all_pairs)


class ImageDataset2(Dataset):
    def __init__(self, csv_file_list,
                 img_dir_list,
                 transform=None,
                 transform2=None,
                 max_train_sample=8125, # kadid10k
                 expansion=3,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = []
        self.max_train_sample = max_train_sample
        for i, item in enumerate(csv_file_list):
            data_item = pd.read_csv(item, sep='\t', header=None)
            #self.data.append()
            db_size = len(data_item)
            repeats = int(np.ceil(self.max_train_sample / db_size) - 1)
            offset = (self.max_train_sample % db_size)
            original_data = data_item
            if repeats > 1:
                for j in range(repeats - 1):
                    data_item = pd.concat([data_item, original_data])
            if offset > 0:
                offset_data = original_data.iloc[:offset, :]
                data_item = pd.concat([data_item, offset_data])
            for _ in range(expansion-1):
                data_item = pd.concat([data_item, data_item])
            self.data.append(data_item)

        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir_list = img_dir_list
        self.transform = transform
        self.transform2 = transform2
        self.loader = get_loader()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        sample = []
        for i, item in enumerate(self.data):
            image_name = os.path.join(self.img_dir_list[i], item.iloc[index, 0])
            I = self.loader(image_name)
            if (i == 4) | (i == 6):
                I = self.transform2(I)
            else:
                I = self.transform(I)

            mos = item.iloc[index, 1]
            std = item.iloc[index, 2]
            sample_item = {'I': I, 'mos': mos, 'std': std}
            sample.append(sample_item)

        return sample

    def __len__(self):
        return len(self.data[0].index)

