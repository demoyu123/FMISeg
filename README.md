# FMISeg
This repo is the official implementation of "**Frequency-domain Multi-modal Fusion for
 Language-guided Medical Image Segmentation**" 
## Prepare
### 1. Environment  
Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. Dataset
QaTa-COV19 Dataset See Kaggle: [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)--**We use QaTa-COV19-v2 in our experiments.**

MosMedData+ Dataset: [Link (Original)](http://medicalsegmentation.com/covid19/)--You need to use 'convert('L')' to convert the label to a grayscale image.

QaTa-COV19 and  MosMedData+ Text Annotations:Check out the related content in LViT: [https://github.com/HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT)

**Thanks to Li et al. for their contributions. If you use this dataset, please cite their work.**

Once the dataset is acquired, you can use the **utils/wave.py** file to perform wave transforms to get both high-frequency and low-frequency images.
### 3. Download the pretrained model of CXR-BERT and ConvNeXt
   We have provided the corresponding pretrained model under the **lib** folder, if you want to get it from the official website, you can get it from the link below:
   
   CXR-BERT-specialized see: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/tree/main  
   ConvNeXt-tiny see: https://huggingface.co/facebook/convnext-tiny-224/tree/main
   

## Train and Test
We use PyTorch Lightning for model training, and you can change the configuration in **config/train.yaml**

For model training, you can use the following commands:  ```python train.py```  
For model testing, you can use the following commands:  ```python evaluate.py```
