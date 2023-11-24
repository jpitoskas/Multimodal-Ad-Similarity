import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from model import *




def class_balanced_random_split(idx2label, seed=None, test_ratio_per_class=0.15):
    """
    Class-balanced dataset split into train and test partitions.
    
    Args:
        idx2label (list): List of labels in the order they appear in the dataset
        seed (int, optional): Random seed (Default: None)
        test_ratio_per_class (float, optional): Percentage of test samples per class (Default: 0.15)

    Returns:
        (tuple):
            * train_indices (list): list of the indices of the train samples
            * test_indices (list): list of the indices of the test samples
    """

    class_indices = {}
    for idx, label in enumerate(idx2label):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)


    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        if len(indices) > 1:
            train_idx, test_idx = train_test_split(indices, test_size=test_ratio_per_class, random_state=seed)
        else:
            train_idx, test_idx = indices.copy(), []
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    return train_indices, test_indices



def get_positive_pairs_indices(pair_ad_dataset, pair_labels):
    """
    Helper function to get the positive pairs indices. 
    Note: discards identical pairs
    """
    
    # Get indices of positive pairs
    positive_pairs_indices = ((pair_labels==1).nonzero(as_tuple=True)[0])
    
    # Condition to discard pairs with identical items
    non_identical_pairs_condition = (positive_pairs_indices // len(pair_ad_dataset.ad_dataset)) != (positive_pairs_indices % len(pair_ad_dataset.ad_dataset))
    
    return positive_pairs_indices[non_identical_pairs_condition]



def get_negative_pairs_indices(pair_ad_dataset, pair_labels):
    """
    Helper function to get the negative pairs indices
    """
    
    # Get indices of negative pairs
    negative_pairs_indices = ((pair_labels==0).nonzero(as_tuple=True)[0])
    
    return negative_pairs_indices



def sample_pair_dataset(pair_ad_dataset, n_pairs_positive=1000, n_pairs_negative=1000):
    """
    Helper function to sample positive and negative example from a PairAdDataset object
    """
    
    
    pair_labels = torch.tensor([pair_ad_dataset.get_label_by_idx(idx) for idx in range(len(pair_ad_dataset))])
    
    # Get indices of positive and negative pairs
    positive_pairs_indices = get_positive_pairs_indices(pair_ad_dataset, pair_labels)
    negative_pairs_indices = get_negative_pairs_indices(pair_ad_dataset, pair_labels)

    
    # How many pairs from each class
    n_pairs_positive = min(n_pairs_positive, len(positive_pairs_indices))
    n_pairs_negative = min(n_pairs_negative, len(negative_pairs_indices))


    # Initiate the sampling
    positive_random_samples = torch.randint(0, positive_pairs_indices.size(0), (n_pairs_positive,))
    negative_random_samples = torch.randint(0, negative_pairs_indices.size(0), (n_pairs_negative,))


    # Apply the sampling
    positive_pairs_indices_sampled = positive_pairs_indices[positive_random_samples].tolist()
    negative_pairs_indices_sampled = negative_pairs_indices[negative_random_samples].tolist()

    
    # Concatenate positive and negative indices for new Subset dataset (balanced undersampled dataset)
    pairs_indices_sampled = positive_pairs_indices_sampled + negative_pairs_indices_sampled


    
    pair_ad_dataset_sampled = Subset(pair_ad_dataset, pairs_indices_sampled)
    
    return pair_ad_dataset_sampled


def get_param_groups(model, model_type):
    """
    Function to return the parameter groups of a MultiModalSiameseNetwork object.

    Args:
        model (MultiModalSiameseNetwork): the Multi-Modal Siamese Network
        model_type (str): the Multi-Modal model type (e.g. 'clip' for CLIP)

    Returns:
        (list): the list of the parameter groups
    """

    if not isinstance(model, MultiModalSiameseNetwork):
        raise ValueError("model should be of type MultiModalSiameseNetwork")


    match model_type:
        case "clip":
            param_groups = [
                        {'params': model.multimodal_network.clip_model.text_model.parameters(), 'name': 'clip_text_model'},
                        {'params': model.multimodal_network.clip_model.vision_model.parameters(), 'name': 'clip_visual_model'},
                        {'params': model.multimodal_network.clip_model.text_projection.parameters(), 'name': 'clip_text_projection'},
                        {'params': model.multimodal_network.clip_model.visual_projection.parameters(), 'name': 'clip_visual_projection'}
                        ]
        case _:
            raise NotImplementedError(f"Model type '{model_type}' is not yet supported")
    
    return param_groups



def get_param_groups_for_finetuning(model, model_type):
    """
    Function to return the parameter groups of a MultiModalSiameseNetwork object for fine-tuning.

    Args:
        model (MultiModalSiameseNetwork): the Multi-Modal Siamese Network
        model_type (str): the Multi-Modal model type (e.g. 'clip' for CLIP)

    Returns:
        (list): the list of the parameter groups for fine-tuning
    """

    if not isinstance(model, MultiModalSiameseNetwork):
        raise ValueError("model should be of type MultiModalSiameseNetwork")


    match model_type:
        case "clip":
            param_groups = get_param_groups(model, model_type)[-2:]
        case _:
            raise NotImplementedError(f"Model type '{model_type}' is not yet supported")
    
    return param_groups




