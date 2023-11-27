from utils import *
from model import *

from pathlib import Path
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import torch
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, CLIPProcessor
import os
import argparse


def predict_ad_pair(text_filepath1, text_filepath2, image_filepath1, image_filepath2,
                    load_model_id=4,
                    pretrained_model_name='openai/clip-vit-base-patch32',
                    inference_similarity_threshold=0.905,
                    seed=1
                    ):
    
    reset_seeds(seed)
    
    text_filepath1 = Path(text_filepath1)
    text_filepath2 = Path(text_filepath2)
    
    image_filepath1 = Path(image_filepath1)
    image_filepath2 = Path(image_filepath2)
    

    with open(text_filepath1, 'r') as f:
        text1 = f.read()
    
    with open(text_filepath2, 'r') as f:
        text2 = f.read()
        
    with Image.open(image_filepath1) as img:
        image1 = transforms.ToTensor()(img).unsqueeze(0)
        
        
    with Image.open(image_filepath2) as img:
        image2 = transforms.ToTensor()(img).unsqueeze(0)
        
        
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    # Dirs
    current_dir = Path().absolute()
    working_dir = current_dir.parent

    model_dir_prefix = "Model_"
    experiments_dir = working_dir/'experiments'/pretrained_model_name.replace('/','_')


    # Image Processor: 
    # Preprocess the input image the way the pretrained model expects 
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)

    # Modify 'do rescale' because our images are already scaled to [0,1]
    image_processor_dict = image_processor.to_dict()
    image_processor_dict['do_rescale'] = False
    image_processor = CLIPImageProcessor(**image_processor_dict)


    # CLIPTokenizerFast: 
    # Preprocess/Tokenize the input document the way the pretrained model expects 
    tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model_name)


    # CLIPProcessor: 
    # Combine the two processors into one
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)


    # CLIPModel
    clip_model = CLIPModel.from_pretrained(pretrained_model_name).to(device)
    
    
    # Same as the CLIPModel but instead forward() only returns the image and text embeddings
    multimodal_network = CLIPModelModified(clip_model).to(device)

    # Siamese Netework
    model = MultiModalSiameseNetwork(multimodal_network).to(device)

    
    if load_model_id is not None:
        # Load weights from fine-tuning
        load_dir = os.path.join(experiments_dir, model_dir_prefix + str(load_model_id))
        load_path = os.path.join(load_dir, f"checkpoint_{load_model_id}.pt")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    
    
    
    model.eval()
    with torch.no_grad():

        image1 = image1.to(device)
        image2 = image2.to(device)

        inputs1 = processor(text=text1, images=image1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = processor(text=text2, images=image1, return_tensors="pt", padding=True, truncation=True)

        # Move tensors to the device
        inputs1 = {key: value.to(device) for key, value in inputs1.items()}
        inputs2 = {key: value.to(device) for key, value in inputs2.items()}

        outputs1, outputs2 = model(inputs1, inputs2)

        cosine_similarity = F.cosine_similarity(outputs1, outputs2)
    
    if cosine_similarity.item() > inference_similarity_threshold:
        print("Similar")
        return 1
    else:
        print("Dissimilar")
        return 0
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ad Pair Prediction')
    parser.add_argument('--text_filepath1', type=str, help='filepath of first ad title', required=True)
    parser.add_argument('--text_filepath2', type=str, help='filepath of second ad title', required=True)
    parser.add_argument('--image_filepath1', type=str, help='filepath of first ad thumbnail', required=True)
    parser.add_argument('--image_filepath2', type=str, help='filepath of second ad thumbnail', required=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--load_model_id', type=int, default=4)
    parser.add_argument('--inference_similarity_threshold', type=float, default=0.905, help='Similarity decision threshold. Used only with inferece=True')
    parser.add_argument('--pretrained_model_name', type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    kwargs = vars(args)
    predict_ad_pair(**kwargs)



