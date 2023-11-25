import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='DeepLab-AdSimilarity')

    
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',  help='number of epochs to train (default: 100)')
    
    # parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='number of warmup training epochs (default: 10)')
    # parser.add_argument('--warmup_start_factor', type=float, default=0.1, help='factor of the base lr to determine initial warmup lr (default: 0.1)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay coefficient ')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 coefficient')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 coefficient')
    # parser.add_argument('--scheduler', default='multistep', choices=['multistep', 'cosine', 'linear', 'exponential'], help='scheduler (multistep, cosine, linear, exponential)')
    # parser.add_argument('--milestones', nargs='+', type=float, default=[0.5, 0.8, 0.9], help="list of scheduler's milestones (floats between 0 and 1), only when scheduler='multistep'")
    # parser.add_argument('--llrd_factor', type=float, default=0.9, help="layer-wise learning rate decay factor (default: 0.9)")
    parser.add_argument('--inference', action='store_true', default=False, help='inference mode (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--backbone_model_config_name', default='microsoft/swin-tiny-patch4-window7-224', help='backbone model configuration name')
    # parser.add_argument('--load_model_id', type=int, default=None)
    # parser.add_argument('--load_backbone_id', type=int, default=None)
    # parser.add_argument('--fpn_out_channels', type=int, default=128, help='FPN output channels (default: 128)')
    # # parser.add_argument('--num_classes', type=int, default=19, help='number of object categories, including background (default: 19)')
    # parser.add_argument('--use_supercategories', action='store_true', default=False, help='group similar categories to supercategories')
    # parser.add_argument('--supercategories_json', default='supercategories_1.json', help='JSON file that groups similar categories to supercategories')
    # parser.add_argument('--loss_weights', nargs=5, metavar=('W_clf', 'W_box_reg', 'W_mask', 'W_objectness', 'W_rpn_box_reg'), type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0],
    #                     help='weights for each mask-rcnn loss component in the following order ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg")')
    # parser.add_argument('--image_mean', nargs=3, metavar=('mu1', 'mu2', 'mu3'), type=float, default=[0.15346707, 0.15595975, 0.20085395], help='pixel mean per channel for data normalization')
    # parser.add_argument('--image_std', nargs=3, metavar=('std1', 'std2', 'std3'), type=float, default=[0.14982773, 0.13622056, 0.16446136], help='pixel mean per channel for data normalization')
    # parser.add_argument('--box_size_by', default='height', choices=['height', 'width'], help='height or width')
    # parser.add_argument('--image_size', nargs=2, metavar=('H', 'W'), default=(512, 512), type=int, help='input image size (height, width)')
    # parser.add_argument('--loaded_image_size', nargs=2, metavar=('H', 'W'), default=(512, 512), type=int, help='loaded image size (height, width)')
    # parser.add_argument('--mask_threshold', type=float, default=0.4)
    # parser.add_argument('--score_threshold', type=float, default=0.05)
    # parser.add_argument('--excluded_categories', nargs='+', type=str, default=[], help='pixel mean per channel for data normalization')
    parser.add_argument('--num_workers', type=int, default=1)


    parser.add_argument('--n_pairs_train', type=int, default=10000)
    parser.add_argument('--n_pairs_val', type=int, default=2500)
    parser.add_argument('--n_pairs_test', type=int, default=2500)
    parser.add_argument('--positive_percentage_train', type=float, default=0.5)
    parser.add_argument('--positive_percentage_val', type=float, default=0.5)
    parser.add_argument('--positive_percentage_test', type=float, default=0.5)
    
    # parser.add_argument('--model_type', type=str, default="clip")
    parser.add_argument('--pretrained_model_name', type=str, default="openai/clip-vit-base-patch32")


    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive Loss margin hyperparameter ')

    args = parser.parse_args()

    return args