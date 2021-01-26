# Tasks
1. Experiment with dice loss, focal loss
2. Add image augmentation for training (classifications and segmentations)
3. Add Conditional Random Field for post processing segmentation
4. Achieve 82% dice score on segmentation dataset
4. Add demo notebooks and docstrings to segmentation pipeline

Results:
1. Focal loss doesn't improve the result, dice loss improve about 1%
2. Augmentation implemented: flip, crop, transpose, brightness and contrast, clahe
3. CRF doesn't improve result
4. Best result: 83.69% dice score on efficientnetB0 - BCE- augmentation
5. Notebook added in segmentation folder