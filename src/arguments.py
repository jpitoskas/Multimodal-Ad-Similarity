import argparse

def get_args_parser():

    parser = argparse.ArgumentParser(description='DeepLab-AdSimilarity')

    
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',  help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay coefficient ')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 coefficient')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2 coefficient')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam epsilon')
    parser.add_argument('--inference', action='store_true', default=False, help='inference mode (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--load_model_id', type=int, default=None)
    parser.add_argument('--fbeta', type=float, default=0.75, help='beta for F-beta score (default: 0.75)')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--evaluation_metric', type=str, default='f1_score')
    parser.add_argument('--inference_similarity_threshold', type=float, default=0.905, help='Similarity decision threshold. Used only with inferece=True')
    parser.add_argument('--n_pairs_train', type=int, default=10000)
    parser.add_argument('--n_pairs_val', type=int, default=2500)
    parser.add_argument('--n_pairs_test', type=int, default=2500)
    parser.add_argument('--positive_percentage_train', type=float, default=0.5)
    parser.add_argument('--positive_percentage_val', type=float, default=0.5)
    parser.add_argument('--positive_percentage_test', type=float, default=0.5)
    
    parser.add_argument('--pretrained_model_name', type=str, default="openai/clip-vit-base-patch32")


    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive Loss margin hyperparameter ')


    return parser