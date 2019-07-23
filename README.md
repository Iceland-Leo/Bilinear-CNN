# Bilinear-CNN
A pytorch implementation of bilinear CNN for fine-grained image recognition in paper [Bilinear CNNs for Fine-grained Visual
Recognition](https://arxiv.org/pdf/1504.07889.pdf) by Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.  
The framework of the entire model is shown in the figure below:
<div align=center><img src="framework.png" width="600" height="300"/></div>  
**In this implementation, the base network uses resnet34 structure**

## Dependencies ##  
python >= 3.5  
pytorch >= 0.4  
In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:  
- `tqdm`  

## Data ##  
Download the [CUB_birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) images and annotations. Extract them to `data/raw/`  

## Training ##  
According to the requirements of this paper, the training of this model is divided into the following two steps:  
- The parameters of the pre-training model are fixed, only the last full connection layer is trained. Run the following code.  
  `python bilinear_ResNet_linear_layer.py`  
  **In this step, the mean value and variance of images data need to be calculated for image preprocessing**
- All parameters in the model are trained. Run the following code.  
  `python bilinear_ResNet_fine_tuning.py`  

## Results ##  
In this implementation, the accuracy of the model on the test dataset of the CUB_bird can reach about 83%. If you want get the model file, please contact me via email:785091715@qq.com
