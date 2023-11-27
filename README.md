# DeepLab Ad Similarity


[Official Pytorch implementation]

A repo with code for Ad Deduplication using pretrained CLIP given an ad title and thumbnail. 


**Author**: Giannis Pitoskas \
**E-mail**: [jpitoskas@gmail.com](mailto:jpitoskas@gmail.com)



## Instructions:

We encourage to create a virtual environment and install the project's dependencies.

### For the project dependencies:
```
pip install -r requirements.txt
```

### Train with defaults
```
python src/main.py
```



### Inference with defaults
```
python src/main.py --inference --load_model_id [model_id]
```

### Arguments

`--batch_size 1`<br>
`--n_epochs 100`<br>
`--lr 0.0005`<br>
`--weight_decay 0.01`<br>
`--beta1 0.9`<br>
`--beta2 0.999`<br>
`--adam_epsilon 1e-08`<br>
`--inference`<br>
`--no_cuda`<br>
`--seed 1`<br>
`--load_model_id None`<br>
`--fbeta 0.5`<br>
`--num_workers 1`<br>
`--evaluation_metric precision`<br>
`--inference_similarity_threshold 0.86`<br>
`--n_pairs_train 10000`<br>
`--n_pairs_val 2500`<br>
`--n_pairs_test 2500`<br>
`--positive_percentage_train 0.5`<br>
`--positive_percentage_val 0.5`<br>
`--positive_percentage_test 0.5`<br>
`--pretrained_model_name openai/clip-vit-base-patch32`<br>
`--margin 1.0`<br>

### Example custom run for training
```
python src/main.py --n_epochs 10 --lr 5e-3 --seed 42
```


### Example custom run for inference
```
python src/main.py --inference --load_model_id [model_id] --batch_size 64 --inference_similarity_threshold f1_score
```



## Dataset:

You can download the Ad Dataset [here](https://storage.googleapis.com/deeplab/projects/dedup/dataset.zip).

Unzip `dataset.zip` and follow this structure:

- for the text data: `data/dataset/data.txt`
- for the image data: `data/dataset/images`

