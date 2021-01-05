# Tasks
1. (Optional) Add docstrings and demos to classifier module.
2. Extend current pipeline to segmentation problem.
3. Download dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
4. Create a script to split train/valid/test at ratio 7/1/2, make sure every portion as all the tumor classes.
5. Build a U-net model and train a first baseline on the dataset. Treat all tumor classes as tumor, make it a binary segmentation problem. Metric: Dice-score.
6. (Optional) Create a module that automatically create a U-net based model that can accept different backbones in its encoder (VGG, Resnet, Densenet, Efficientnet, ...).

#Result
1. Demo notebook link: classifier/internship.ipynb