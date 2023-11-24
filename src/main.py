#!/usr/bin/env python3

from dataloaders import *
from arguments import *
from model import *

from pathlib import Path
import sys
import torch
import logging
import random
import numpy as np


from transformers import AutoProcessor, CLIPModel

from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm



current_dir = Path().absolute()
# if str(current_dir.parent/'src') not in sys.path:
#     sys.path.append(str(current_dir.parent/'src'))
    
from dataloaders import *




if __name__ == '__main__':

    # Parse arguments
    args = parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



    working_dir = current_dir.parent
    data_dir = working_dir.joinpath('data', 'dataset')

    text_data_filepath = data_dir.joinpath('data.txt')
    thumbnail_data_dir = data_dir.joinpath('images')


    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)
                   ])
        #  handlers=[logging.FileHandler(os.path.join(new_model_dir, f'trainlogs_{new_id}.log'))
        #            ])


    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        logging.info(f"\nDevice: \n- {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(args.seed)
    else:
        logging.info(f"\nDevice: {'CPU'}")





    logging.info("\nArguments:")
    logging.info('\n'.join(f'- {k}: {v}' for k, v in vars(args).items()))
    logging.info('\n'.join(f'args.{k}={v}' for k, v in vars(args).items()))
    # exit()



    logging.info("\n\nLoading Dataset...")

    pair_train_loader, pair_test_loader = get_pair_dataloaders_combined(args, text_data_filepath, thumbnail_data_dir)
    logging.info(f'Train: {len(pair_train_loader.dataset)} - Test: {len(pair_test_loader.dataset)}')
    

    

    # Model
    match args.model_type:
        case "clip":
            processor = AutoProcessor.from_pretrained(args.pretrained_model_name)
            clip_model = CLIPModel.from_pretrained(args.pretrained_model_name).to(device)
            multimodal_network = CLIPModelModified(clip_model).to(device)
        case _:
            raise NotImplementedError(f"Model type '{args.model_type}' is not yet supported")
    
    model = MultiModalSiameseNetwork(multimodal_network).to(device)


    # Loss
    loss_fn = ContrastiveLoss(margin=args.margin)


    # Optimizer
    param_groups = get_param_groups_for_finetuning(model, args.model_type)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)


     # Trainable parameters
    trainable_params = sum(p.numel() for p in optimizer.param_groups[0]['params'] if p.requires_grad)
    logging.info(f'\nTrainable parameters: {trainable_params:,}\n')




    # Train function
    def train(epoch, pair_loader, model, processor, loss_fn, optimizer, device):

        model.train()
        # train_loss = 0
        for _, (data, targets) in enumerate(tqdm(pair_loader)):

            (text1, image1), (text2, image2) = data
     
            image1 = image1.to(device)
            image2 = image2.to(device)

            targets = targets.to(device)
            
            inputs1 = processor(text=text1, images=image1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = processor(text=text2, images=image1, return_tensors="pt", padding=True, truncation=True)

            # Move tensors to the device
            inputs1 = {key: value.to(device) for key, value in inputs1.items()}
            inputs2 = {key: value.to(device) for key, value in inputs2.items()}



            optimizer.zero_grad()

            outputs1, outputs2 = model(inputs1, inputs2)

            loss = loss_fn(outputs1, outputs2)
            # train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            


        # train_loss /= len(pair_loader.dataset)

        # return train_loss
        return







