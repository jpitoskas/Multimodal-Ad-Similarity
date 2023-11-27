# DeepLab Ad Similarity


[Official Pytorch implementation]

A repo with code for Ad Deduplication using pretrained CLIP given an ad title and thumbnail. 


**Author**: Giannis Pitoskas <br>
**E-mail**: [jpitoskas@gmail.com](mailto:jpitoskas@gmail.com)



## Instructions:

We encourage to create a virtual environment and install the project's dependencies.

### Install Dependencies:
```
pip install -r requirements.txt
```

### Train with Default Arguments
```
python src/main.py
```



### Inference with Default Arguments
```
python src/main.py --inference --load_model_id [model_id]
```

### Arguments

`--batch_size 1`<br>
`--n_epochs 5`<br>
`--lr 5e-4`<br>
`--weight_decay 0.2`<br>
`--beta1 0.9`<br>
`--beta2 0.98`<br>
`--adam_epsilon 1e-6`<br>
`--inference`<br>
`--no_cuda`<br>
`--seed 1`<br>
`--load_model_id None`<br>
`--fbeta 0.75`<br>
`--num_workers 1`<br>
`--evaluation_metric f1_score`<br>
`--inference_similarity_threshold 0.905`<br>
`--n_pairs_train 10000`<br>
`--n_pairs_val 2500`<br>
`--n_pairs_test 2500`<br>
`--positive_percentage_train 0.5`<br>
`--positive_percentage_val 0.5`<br>
`--positive_percentage_test 0.5`<br>
`--pretrained_model_name openai/clip-vit-base-patch32`<br>
`--margin 1.0`<br>

### Example Custom Run for Training
```
python src/main.py --n_epochs 10 --lr 5e-3 --seed 42
```


### Example Custom Run for Inference
```
python src/main.py --inference --load_model_id [model_id] --batch_size 64 --inference_similarity_threshold f1_score
```



## Dataset:

You can download the Ad Dataset [here](https://storage.googleapis.com/deeplab/projects/dedup/dataset.zip).

Unzip `dataset.zip` and follow this structure:

- for the text data: `data/dataset/data.txt`
- for the image data: `data/dataset/images`


## Download our Pre-trained Model Checkpoint:

You can download our pre-trained model checkpoint from the following link:

[Download Model Checkpoint (.pt file)](https://drive.google.com/file/d/1-9A89LmE-OfI-0KWPnVgkQbdXL7P0epX/view?usp=sharing)

### Usage Instructions


After downloading the model checkpoint, you can:

- use it for further training/fine-tuning:

```
python src/main.py --inference --load_model_id [model_id]
```

- use it for inference:

```
python src/main.py --inference
```
