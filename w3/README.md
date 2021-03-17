# Tasks
1. (Optional) Add docstrings and demos to classifier module.
2. Extend current pipeline to segmentation problem.
3. Download dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
4. Create a script to split train/valid/test at ratio 7/1/2, make sure every portion as all the tumor classes.
5. Build a U-net model and train a first baseline on the dataset. Treat all tumor classes as tumor, make it a binary segmentation problem. Metric: Dice-score.
6. (Optional) Create a module that automatically create a U-net based model that can accept different backbones in its encoder (VGG, Resnet, Densenet, Efficientnet, ...).

#Result
1. Demo notebook link: classifier/internship.ipynb
6. Backbone version1: code vgg16, resnet18, resnet101


- Investigated the deep learning methods for image classification and segmentation.
- Built Pytorch training pipeline for the two problems.
- Evaluated the effects of models, optimizers, and losses, learning rate scheduler on the performance of the methods.
- Learn preprocess, augment data, process imbalance data and visualize in training process
- Add attention, OCR modules to existence model
- Observations: 
+ Efficient and densenet backbones is better than resnet, Unet is good for segmentation in trained datasets.
+ For imbalance data: train with high positive rate examples first, after decrease; some models need weighted loss
+ LRScheduler, Augmentation, attention, OCR modules make result better
