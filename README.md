# Multimodal Variational Autoencoder
A PyTorch implementation of *Multimodal Generative Models for Scalable Weakly-Supervised Learning* (https://arxiv.org/abs/1802.05335). Code adapted from [original implementation](https://github.com/mhw32/multimodal-vae-public) by authors.

## Setup/Installation

Open a new conda environment and install the necessary dependencies. See [here](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) for more details on installing `dlib`.
```
conda create -n multimodal python=2.7 anaconda
# activate the environment
source activate multimodal

# install the pytorch
conda install pytorch torchvision -c pytorch

pip install tqdm
pip install scikit-image
pip install python-opencv
pip install imutils

# install dlib
brew install cmake
brew install boost
pip install dlib
```


### MNIST 
Treat images as one modality and the label (integer 0 to 9) as a second. 

```
cd mnist
CUDA_VISIBLE_DEVICES=0 python train.py --lambda-text 50. --cuda
# model is stored in ./trained_models
CUDA_VISIBLE_DEVICES=0 python sample.py ./trained_models/model_best.pth.tar --cuda
# you can also condition on the label
CUDA_VISIBLE_DEVICES=0 python sample.py ./trained_models/model_best.pth.tar --condition-on-text 5 --cuda
```
