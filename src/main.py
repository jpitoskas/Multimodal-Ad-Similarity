#!/usr/bin/env python3

from dataloaders import *
from arguments import *

from pathlib import Path
import sys
import torch
import logging
import random
import numpy as np



from torchvision import transforms



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



    # class EmptyClass:
    #     pass

    # # Instantiating the EmptyClass
    # args = EmptyClass()
    # args.batch_size = 128
    # args.num_workers = 0
    # args.cuda = True


    logging.info("\nArguments:")
    logging.info('\n'.join(f'- {k}: {v}' for k, v in vars(args).items()))




    logging.info("\n\nLoading Dataset...")   
    train_loader_combined, test_loader_combined = get_dataloaders_combined(args, text_data_filepath, thumbnail_data_dir)

    logging.info(f'Train: {len(train_loader_combined.dataset)} - Test: {len(test_loader_combined.dataset)}')


    
    for i, ((text, img), label) in enumerate(train_loader_combined):
        break

        if i % 10 == 0:
            break
            print(i)



