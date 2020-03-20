## FCN-8s implementation in TensorFlow

### Preface
This repository is forked from **pierluigiferrari/fcn8s_tensorflow.** 
Contribution to the existing repository

* [x] Training script 
* [x] Modularize network script into feature extractor and application network 
* [ ] Add multiple core networks [WIP]
    * [x] GoogleNet
    * [ ] Inception v2
    * [ ] Inception v3
    * [ ] Resnet50
    * [ ] MobileNet
    * [ ] Squeezenet
    
### Training
1. Create an environment with python packages from requirements file
2. [Download pre-trained VGG-16 model](https://drive.google.com/open?id=0B0WbA4IemlxlWlpBd2NBeFUteEE)
3. Add cityscapes dataset paths and pretrained weights path in train_config.py
4. Start Training

