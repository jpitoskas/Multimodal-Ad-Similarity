import csv
from pathlib import Path
import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, random_split


from utils import *


class AdTitleDataset(Dataset):
    """
    Custom PyTorch dataset to handle ad title strings (sentences).
    """
    
    
    def __init__(self, filepath, transforms=None, shuffle=False):
        """
        Initialize the AdTitleDataset
        
        Args:
          filepath (str): Path to the .txt file containing ad information
          transforms (callable, optional): Transformations to apply to the data.
          shuffle (bool, optional): Whether to shuffle the dataset (Default: False).
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
    
    
    def get_label_by_idx(self, idx):

        item_id = self.ids[idx]
        info =  self.data[item_id]
        label = info['label']

        return label
    






class AdThumbnailDataset(Dataset):
    """
    Custom PyTorch dataset to handle ad thumbnails (images).
    """
    
    def __init__(self, imgs_dir, transforms=None, shuffle=False):
        """
        Initialize the AdThumbnailDataset
        
        Args:
          imgs_dir (str): Directory where the ad thumbnail image files are stored
          transforms (callable, optional): Transformations to apply to the data
          shuffle (bool, optional): Whether to shuffle the dataset (Default: False)
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

        if img.mode != 'RGB': 
            # There are Grayscale images in the dataset that we need to convert to a consistent format => RGB
            img = img.convert('RGB')


        label = item['label']    
    
        if self.transforms:
            img = self.transforms(img)
        
        return img, label


    def __len__(self):
        """
        Returns the length of the image dataset
        (the total number of ad thumbnails).
        
        Returns:
            (int): the length of the image dataset
        """
        return len(self.ids)

    def get_info_by_idx(self, idx):
        """
        Returns a dictionary corresponding to the given
        index. This dictionary contains information 
        about the ad as {cc, label, filename}.
        
        Args:
            idx (int): index of the item in the dataset
            
        Returns:
            (dict):
                * 'cc' (int): ad id
                * 'label' (int): class id
                * 'filename' (str): thumbnail filename
        """

        item_id = self.ids[idx]
        return self.data[item_id]


    def get_label_by_idx(self, idx):

        item_id = self.ids[idx]
        info =  self.data[item_id]
        label = info['label']

        return label



class CombinedAdDataset(Dataset):
    """
    Custom PyTorch dataset combining AdTitleDataset and AdThumbnailDataset into one.
    """
    

    def __init__(self, 
                 ad_title_dataset: AdTitleDataset,
                 ad_thumbnail_dataset: AdThumbnailDataset,
                 shuffle=False):
        """
        Initialize the CombinedAdDataset
        
        Args:
            ad_title_dataset (AdTitleDataset): Dataset containing ad title information.
            ad_thumbnail_dataset (AdThumbnailDataset): Dataset containing ad thumbnail images.
            shuffle (bool, optional): Whether to shuffle the dataset (Default: False).
        """

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        
        if not isinstance(ad_title_dataset, AdTitleDataset):
            raise TypeError("ad_title_dataset should be an instance of AdTitleDataset")
        
        if not isinstance(ad_thumbnail_dataset, AdThumbnailDataset):
            raise TypeError("ad_thumbnail_dataset should be an instance of AdThumbnailDataset")
        
        
        self.ad_title_dataset = ad_title_dataset
        self.ad_thumbnail_dataset = ad_thumbnail_dataset
        
        assert len(ad_title_dataset) == len(ad_thumbnail_dataset), "Datasets must have the same length"
        assert set(ad_title_dataset.ids) == set(ad_thumbnail_dataset.ids), "Datasets must correspond to the same items ids"
        assert ad_title_dataset.ids == ad_thumbnail_dataset.ids, "Dataset ids not in the same order. When initializing the two Datasets, set shuffle=False."
        
        
        self.ids = ad_title_dataset.ids.copy()
        
        
        
    def __getitem__(self, idx):
        """
        Get the combined data for a given index.
        
        Args:
            idx (int): Index of the item in the dataset
            
        Returns:
            (tuple):
                * text (string): raw text input
                * img (PIL Image or Tensor): transformed image
                * label (int): class id    
        """
        
        text, label_text = self.ad_title_dataset[idx]
        img, label_img = self.ad_thumbnail_dataset[idx]
        
        assert label_text == label_img, f'Label mismatch between the two datasets on the item with id: {self.ids[idx]}' 
        
        label = label_text # Assigning one of the labels, knowing they are equal
        
        # import torch
        # return ('text', torch.tensor([17])), 42
        return (text, img), label
    

    def get_label_by_idx(self, idx):

        text_info = self.ad_title_dataset.get_info_by_idx(idx)
        label_text = text_info['label']

        img_info = self.ad_thumbnail_dataset.get_info_by_idx(idx)
        label_img = img_info['label']

        assert label_text == label_img, f'Label mismatch between the two datasets on the item with id: {self.ids[idx]}' 

        label = label_text # Assigning one of the labels, knowing they are equal

        return label


    def __len__(self):
        """
        Returns the length of the ad dataset.
        
        Returns:
            (int): the length of the ad dataset
        """
        return len(self.ids)
    


from torch.utils.data import Dataset


class PairAdDataset(Dataset):
    def __init__(self, ad_dataset):
        
        self.ad_dataset = ad_dataset
        self.ad_dataset_size = len(self.ad_dataset)
        
        if isinstance(self.ad_dataset, torch.utils.data.dataset.Subset):
            self.original_indices = self.ad_dataset.indices
            self.ad_labels = {idx : self.ad_dataset.dataset.get_label_by_idx(idx) for idx in self.original_indices}            
        else:
            self.original_indices = list(range(self.ad_dataset_size))
            self.ad_labels = {idx : self.ad_dataset.get_label_by_idx(idx) for idx in range(len(self.ad_dataset))}

            
            
    
    def __getitem__(self, idx):
        
        i =  idx // self.ad_dataset_size
        j = idx % self.ad_dataset_size
        
        original_i = self.original_indices[i]
        original_j = self.original_indices[j]

        if isinstance(self.ad_dataset, torch.utils.data.dataset.Subset):
            (text1, img1), _ = self.ad_dataset.dataset[original_i]
            (text2, img2), _ = self.ad_dataset.dataset[original_j]
        else:
            (text1, img1), _ = self.ad_dataset[original_i]
            (text2, img2), _ = self.ad_dataset[original_j]
        
        p1 = (text1, img1)
        p2 = (text2, img2)
        pair = (p1, p2)

        pair_label = int(self.ad_labels[original_i] == self.ad_labels[original_j])
        
        return pair, pair_label
    
    

    def __len__(self):
        return self.ad_dataset_size**2
    

    
    def get_label_by_idx(self, idx):
        
        i =  idx // self.ad_dataset_size
        j = idx % self.ad_dataset_size
        
        original_i = self.original_indices[i]
        original_j = self.original_indices[j]
        
        pair_label = int(self.ad_labels[original_i] == self.ad_labels[original_j])
        
        return pair_label
        


    


def get_datasets_combined(args,
                          text_data_filepath,
                          thumbnail_data_dir
                          ):

    img_transf = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std)
            ])
    
    ad_title_dataset = AdTitleDataset(text_data_filepath)
    ad_thumbnail_dataset = AdThumbnailDataset(thumbnail_data_dir, transforms=img_transf)

    combined_ad_dataset = CombinedAdDataset(ad_title_dataset, ad_thumbnail_dataset)

    idx2label = [combined_ad_dataset.get_label_by_idx(idx) for idx in range(len(combined_ad_dataset))]
    combined_ad_dataset.train_indices, combined_ad_dataset.val_indices, combined_ad_dataset.test_indices = class_balanced_random_split(idx2label, seed=args.seed)


    
    # Create Subset for each split
    train_dataset_combined = Subset(combined_ad_dataset, combined_ad_dataset.train_indices)
    val_dataset_combined = Subset(combined_ad_dataset, combined_ad_dataset.val_indices)
    test_dataset_combined = Subset(combined_ad_dataset, combined_ad_dataset.test_indices)

    return train_dataset_combined, val_dataset_combined, test_dataset_combined




def get_dataloaders_combined(args,
                             text_data_filepath,
                             thumbnail_data_dir
                             ):
    
    """
    Function to create Dataloaders for both train and test splits of a CombinedAdDataset instance

    Args:
        args (argparse.Namespace): arguments
        text_data_filepath (str): Path to the .txt file containing ad information
        thumbnail_data_dir: (str): Directory where the ad thumbnail image files are stored

    Returns:
        (tuple):
            * train_loader_combined (torch.utils.data.dataloader.DataLoader): Dataloader for the train split
            * val_loader_combined (torch.utils.data.dataloader.DataLoader): Dataloader for the val split
            * test_loader_combined (torch.utils.data.dataloader.DataLoader): Dataloader for the test split
    """

    
    train_dataset_combined, val_dataset_combined, test_dataset_combined = get_datasets_combined(args, text_data_filepath, thumbnail_data_dir)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    # Create DataLoader for each split
    train_loader_combined = DataLoader(train_dataset_combined, batch_size=args.batch_size, shuffle=True)
    val_loader_combined = DataLoader(val_dataset_combined, batch_size=args.batch_size, shuffle=True)
    test_loader_combined = DataLoader(test_dataset_combined, batch_size=args.batch_size, shuffle=False)


    return train_loader_combined, val_loader_combined, test_loader_combined




def get_pair_dataloaders_combined(args,
                                  text_data_filepath, 
                                  thumbnail_data_dir,
                                  ):
    
    train_dataset_combined, val_dataset_combined, test_dataset_combined = get_datasets_combined(args, text_data_filepath, thumbnail_data_dir)


    pair_ad_dataset_train = PairAdDataset(train_dataset_combined)
    pair_ad_dataset_val = PairAdDataset(val_dataset_combined)
    pair_ad_dataset_test = PairAdDataset(test_dataset_combined)



    n_pairs_positive_train = int(args.positive_percentage_train*args.n_pairs_train)
    n_pairs_negative_train = args.n_pairs_train - n_pairs_positive_train

    n_pairs_positive_val = int(args.positive_percentage_val*args.n_pairs_val)
    n_pairs_negative_val = args.n_pairs_val - n_pairs_positive_val

    n_pairs_positive_test = int(args.positive_percentage_test*args.n_pairs_test)
    n_pairs_negative_test = args.n_pairs_test - n_pairs_positive_test


    pair_ad_dataset_train_sampled = sample_pair_dataset(pair_ad_dataset_train, n_pairs_positive=n_pairs_positive_train, n_pairs_negative=n_pairs_negative_train)
    pair_ad_dataset_val_sampled = sample_pair_dataset(pair_ad_dataset_val, n_pairs_positive=n_pairs_positive_val, n_pairs_negative=n_pairs_negative_val)
    pair_ad_dataset_test_sampled = sample_pair_dataset(pair_ad_dataset_test, n_pairs_positive=n_pairs_positive_test, n_pairs_negative=n_pairs_negative_test)


    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    # Create DataLoader for each split
    pair_train_loader_sampled = DataLoader(pair_ad_dataset_train_sampled, batch_size=args.batch_size, shuffle=True)
    pair_val_loader_sampled = DataLoader(pair_ad_dataset_val_sampled, batch_size=args.batch_size, shuffle=False)
    pair_test_loader_sampled = DataLoader(pair_ad_dataset_test_sampled, batch_size=args.batch_size, shuffle=False)

    return pair_train_loader_sampled, pair_val_loader_sampled, pair_test_loader_sampled

