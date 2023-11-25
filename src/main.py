#!/usr/bin/env python3

from dataloaders import *
from arguments import *
from model import *
from train import *
from evaluate import *

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
import warnings
import os
import datetime

import matplotlib.pyplot as plt
import json




current_dir = Path().absolute()
# if str(current_dir.parent/'src') not in sys.path:
#     sys.path.append(str(current_dir.parent/'src'))
    
from dataloaders import *



warnings.filterwarnings("ignore")



if __name__ == '__main__':

    # Parse arguments
    args = parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False




    if args.inference and args.load_model_id is None:
        raise ValueError("Should load model for inference mode.")
    

    if 'clip' in args.pretrained_model_name:
        args.model_type = 'clip'
    else:
        raise NotImplementedError(f'Unsupported model_type for {args.pretrained_model_name}')



    # Dirs
    working_dir = current_dir.parent
    data_dir = working_dir.joinpath('data', 'dataset')

    text_data_filepath = data_dir.joinpath('data.txt')
    thumbnail_data_dir = data_dir.joinpath('images')



    model_dir_prefix = "Model_"
    experiments_dir = working_dir/'experiments'/args.pretrained_model_name.replace('/','_')
    # Create directories for logs
    if os.path.exists(experiments_dir):
        model_dirs = [model_dir if os.path.isdir(os.path.join(experiments_dir, model_dir)) else None for model_dir in os.listdir(experiments_dir)]
        model_dirs = list(filter(None, model_dirs))
        ids = [int(dd.replace(model_dir_prefix,"")) if (model_dir_prefix) in dd and dd.replace(model_dir_prefix,"").isnumeric() else None for dd in model_dirs]
        ids = list(filter(None, ids))
        new_id = str(max(ids) + 1) if ids else "1"
    else:
        experiments_dir.mkdir(parents=True)
        new_id = "1"

    new_model_dir = experiments_dir.joinpath(model_dir_prefix + new_id)
    new_model_dir.mkdir()


    # Logging
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(new_model_dir, f'trainlogs_{new_id}.log'))                              
                   ])
    
        #  handlers=[logging.FileHandler(os.path.join(new_model_dir, f'trainlogs_{new_id}.log'))
        #            ])
    
    logging.info(f'{model_dir_prefix}ID: {new_id}\n')
    logging.info(datetime.datetime.now())



    # Device
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
    # logging.info('\n'.join(f'args.{k}={v}' for k, v in vars(args).items()))
    # exit()



    logging.info("\n\nLoading Dataset...")
    pair_train_loader, pair_val_loader, pair_test_loader = get_pair_dataloaders_combined(args, text_data_filepath, thumbnail_data_dir)
    logging.info(f'Train: {len(pair_train_loader.dataset)} - Val: {len(pair_val_loader.dataset)} - Test: {len(pair_test_loader.dataset)}')
    

    

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








    experiments_dir = working_dir/'experiments'/args.pretrained_model_name.replace('/','_')
    # Load a previously trained model to train more
    if args.load_model_id:
        load_dir = os.path.join(experiments_dir, model_dir_prefix + str(args.load_model_id))
        load_path = os.path.join(load_dir, f"checkpoint_{args.load_model_id}.pt")

        if not os.path.isfile(load_path):
            raise ValueError(f"Cannot find chesckpoint_{args.load_model_id}.pt in {load_dir}")
        
        logging.info(f'\nLoading {model_dir_prefix}{args.load_model_id} from "{load_path}"\n')
        checkpoint = torch.load(load_path)
        # Loading model, optimizer, epoch, train_losses, val_losses
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_losses = checkpoint['val_metrics']
    else:
        init_epoch = 1
        train_losses = []
        val_losses = []
        val_metrics = []


     # Parallelize if more than 1 GPU
    if torch.cuda.device_count() > 1:
        logging.info(f'\nUsing {torch.cuda.device_count()} GPU(s).\n')
        model = nn.DataParallel(model).to(device)
        

    # Train, Val, Test kwargs
    train_kwargs = {'pair_loader': pair_train_loader,
                    'model': model,
                    'processor': processor,
                    'loss_fn': loss_fn,
                    'optimizer': optimizer,
                    'device': device,
                    }

    val_kwargs = {'pair_loader': pair_val_loader,
                  'model': model,
                  'processor': processor,
                  'loss_fn': loss_fn,
                  'device': device,
                  'thresholds': torch.arange(-1, 1, 0.01),
                  'similarity': 'cosine',
                  'optimization_metric': args.evaluation_metric,
                  }

    test_kwargs = {'pair_loader': pair_test_loader,
                   'model': model,
                   'processor': processor,
                   'loss_fn': loss_fn,
                   'device': device,
                   'similarity': 'cosine',
                   'optimization_metric': args.evaluation_metric,
                   }
    

    best_metric_score, best_epoch = 0, 0
    best_threshold = None

    # TRAINING LOOP
    logging.info(f"Fine-tune model on target task for {args.n_epochs} epochs:")
    for epoch in range(init_epoch, args.n_epochs + init_epoch):
        train_loss = train(epoch, **train_kwargs)
        val_loss, metrics, optimal_threshold = test(**val_kwargs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(metrics[args.evaluation_metric])

        logging.info(
            "\n" + 
            f"Epoch [{epoch}/{args.n_epochs}]: Train Loss: {train_loss:.4f}" + ' | ' + f"Validation Loss: {val_loss:.4f}" + "\n" +
            " "*len(f"Epoch [{epoch}/{args.n_epochs}]: ") + f"Validation Metrics (threshold={optimal_threshold:.2f}): " +
            " | ".join([f'{metric_str.capitalize()}: {metric_score:.4f}' for metric_str, metric_score in metrics.items()]) +
            "\n"
        )
        
        if metrics[args.evaluation_metric] > best_metric_score:
                   
            best_metric_score = metrics[args.evaluation_metric]
            best_threshold = optimal_threshold
            best_epoch = epoch
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_metrics': val_metrics,
                        }, os.path.join(new_model_dir, f"checkpoint_{new_id}.pt"))
            
    
    logging.info(f"\nBest {args.evaluation_metric.capitalize()}: {best_metric_score:.4f} on epoch {best_epoch}.")
    # logging.info(f"\nBest validation loss: {best_val_loss:.4f} on epoch {best_val_epoch}.")
    

    test_loss, metrics, _ = test(**test_kwargs, thresholds=torch.tensor([best_threshold]))
    logging.info(
            "\n" + f"Test Loss: {val_loss:.4f}" + "\n" + 
            f"Test Metrics (threshold={optimal_threshold:.2f}): " +
            " | ".join([f'{metric_str.capitalize()}: {metric_score:.4f}' for metric_str, metric_score in metrics.items()]) +
            "\n"
        )


    with open(os.path.join(new_model_dir, "train_losses.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows([train_losses])

    with open(os.path.join(new_model_dir, "val_losses.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows([val_losses])
    
    with open(os.path.join(new_model_dir, "val_metrics.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows([val_metrics])

    

    
    with open(new_model_dir.joinpath('args.json'), 'w') as f_args:
        json.dump(vars(args), f_args)

    
    # Plot Contrastive Loss
    plt.figure(figsize=(10,7))
    plt.title("Contrastive Loss per Epoch")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(new_model_dir, f'loss_{new_id}.png'))

    # Plot Validation Eval metric
    plt.figure(figsize=(10,7))
    plt.title(f"Validation {args.evaluation_metric.capitalize()}")
    plt.plot(val_metrics)
    plt.xlabel("Epochs")
    plt.ylabel(args.evaluation_metric.capitalize())
    plt.grid()
    plt.savefig(os.path.join(new_model_dir, f'{args.evaluation_metric}_{new_id}.png'))


    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]

