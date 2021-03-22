# Decoupled Cross-Modal BERT

## README
For a better user experience, we recommend you to directly use github repository
github [link](https://github.com/dcb_submission/decoupled_cross_modal_bert)

This project is deveopped based on HuggingFace Transformer [link](https://github.com/huggingface/transformers) and SCAN [link](https://github.com/kuanghuei/SCAN)

## Prerequisite

### Install tensorboard_logger
pip install tensorboard_logger

### Install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## Prepare dataset and models

### Dataset
Flickr30K dataset

We extract bounding box features through a pre-trained faster rcnn provided by [link](https://github.com/peteanderson80/bottom-up-attention)

You can download the pre-computed testing features through [link](https://www.dropbox.com/s/bkgzftnavcub1hs/flickr30k_test_frcnnnew.tar.gz?dl=0) 

You can download the pre-computed training features through [link](https://www.dropbox.com/s/ujrwh675occxfzl/flickr30k_train_frcnnnew.tar.gz?dl=0)

unzip your downloaded files, and move them to data/f30k_precomp

### BERT Model
download the pretrained bert model provided by HuggingFace through this [link](https://www.dropbox.com/s/a20ufjz3145g80z/pytorch_model.bin?dl=0)
move your downloaded pytorch_model.bin file to ./bert fold

## Run Script
### Training
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 256 --num_epochs=70 --lr_update=30 --learning_rate=.00006
### Testing
We provide a trained model [link]() on Flickr30K dataset. The model is without pre-training on CC and SBU datasets.
CUDA_VISIBLE_DEVICES=0,1 python test.py --resume checkpoint.pth.tar 


