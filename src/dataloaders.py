import csv
from pathlib import Path
import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader


class AdTitleDataset(Dataset):
    """
    Custom PyTorch dataset to handle ad title strings (sentences).
    """
    
    
    def __init__(self, filepath, transforms=None, shuffle=True):
        """
        Initialize the AdTitleDataset
        
        Args:
        - filepath (str): Path to the .txt file containing ad information
        - transforms (callable, optional): Transformations to apply to the data.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        """
        
        
        self.filepath = filepath
        self.transforms = transforms
        self.shuffle = shuffle
        
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        self.headers = lines[0].strip().split(',')
        
        assert self.headers == ['cc', 'title', 'turl', 'label']
        
        self.data = []
        for line in lines[1:]:
            split_line = line.strip().split(';')
            if len(split_line) == len(self.headers):
                item = {header: attr for header, attr in zip(self.headers, split_line)}
                self.data.append(item)
            elif len(split_line) > len(self.headers) and split_line[2].startswith('http'):      # TODO: should be improved
                split_line = [split_line[0], split_line[1], ';'.join(split_line[2:-1]), split_line[-1]]
                item = {header: attr for header, attr in zip(self.headers, split_line)}
                self.data.append(item)
                
                
        
        self.data = {int(row['cc']) : { 'cc': int(row['cc']),
                                        'title': row['title'],
                                        'turl': row['turl'],
                                        'label': int(row['label'])} for row in self.data}
        

        
        self.ids = list(self.data.keys())
        
        if self.shuffle:
            random.shuffle(self.ids)
        else:
            self.ids = sorted(self.ids)
            
            

        
    def __getitem__(self, idx):
        """
        Returns the transformed dataset item associated with the given index.
        
        Args:
            idx (int): index of the item in the dataset
        
        Returns:
            (tuple):
                * text (string): raw text input
                * label (int): class id
        """

        item_id = self.ids[idx]
        item = self.data[item_id]
        
        text, label = item['title'], item['label']
        
        if self.transforms:
            # TODO
            raise NotImplementedError
        
        return text, label


    def __len__(self):
        """
        Returns the length of the ad title dataset.
        
        Returns:
            (int): the length of the ad title dataset
        """
        
        return len(self.ids)
    
    
    def get_info_by_id(self, item_id):
        """
        Returns a dictionary corresponding to a specific
        item_id. This dictionary contains information 
        about the ad as {cc, title, turl, label}.
        
        Args:
            item_id (int): id of the item in the dataset
        
        Returns:
            (dict):
                * 'cc' (int): ad id
                * 'title' (int): ad title
                * 'turl' (string): ad thumbnail url
                * 'label' (int): class id
        """
        return self.data[item_id]
    
    def get_info_by_idx(self, idx):
        """
        Returns a dictionary corresponding to the given
        index. This dictionary contains information 
        about the ad as {cc, title, turl, label}.
        
        Args:
            idx (int): index of the item in the dataset
            
        Returns:
            (dict):
                * 'cc' (int): ad id
                * 'title' (int): ad title
                * 'turl' (string): ad thumbnail url
                * 'label' (int): class id
        """

        item_id = self.ids[idx]
        return self.data[item_id]
    






class AdThumbnailDataset(Dataset):
    """
    Class to handle the Ad Thumbnail Dataset.
    """
    
    def __init__(self, imgs_dir, transforms=None, shuffle=True):
        """
        Initialize the AdThumbnailDataset
        
        Args:
        - imgs_dir (str): Directory where the ad thumbnail image files are stored
        - transforms (callable, optional): Transformations to apply to the data
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True
        """
        
        
        self.imgs_dir = imgs_dir
        self.transforms = transforms
        self.shuffle = shuffle
        
        
        all_filenames = list(map(os.path.basename, glob.glob(str(self.imgs_dir.joinpath("*.jpg")))))
        
        
        self.data = {}
        for filename in all_filenames:
            label, item_id, _ = filename.split('_')
            label, item_id = int(label), int(item_id)
            self.data[item_id] = {'cc': item_id, 'label': label, 'filename': filename}
            
        
        self.ids = list(self.data.keys())
        
        if self.shuffle:
            random.shuffle(self.ids)
        else:
            self.ids = sorted(self.ids)
        

        
    def __getitem__(self, idx):
        """
        Returns the transformed ad thumbnail associated with the given index.
        
        Args:
            idx (int): index of the item in the dataset
        
        Returns:
            (tuple):
                * img (PIL Image or Tensor): transformed image
                * label (int): class id
        
        """
        item_id = self.ids[idx]
        item = self.data[item_id]
        
        filename = item['filename']
        filepath = self.imgs_dir.joinpath(filename)
        
        img = Image.open(filepath)
        label = item['label']    
    
        if self.transforms:
            raise NotImplementedError
        
        return img, label


    def __len__(self):
        """
        Returns the length of the image dataset
        (the total number of ad thumbnails).
        
        Returns:
            (int): the length of the image dataset
        """
        return len(self.ids)




