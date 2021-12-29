# Diagnosis of recurrence of nasopharyngeal carcinoma

## rNPC detection and classification
============================

### Training
python main_ddp.py

### Inference
python inference_cls.py

### Result of classification (AUC)

| Network       | test1       | test2       | test3       | test4       | guangzhou   | zhongshan   | zhuhai      | foshan      |
| ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|    det+cls    | 0.913       | 0.941       | 0.905       | 0.871       | 0.897       | 0.870       | 0.844       | 0.928       |

## rNPC 2.5D VNet segmentation
============================

This implements training of VNet networks from [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) by F. Milletari, et. al.

### Training
python main_seg.py

### Inference
python inference_seg.py

### Comparative Experiment of segmentation (DSC)

| Network       | Val         | Test1       | SYSUCC      | External    |
| ------------- | ----------- | ----------- | ----------- | ----------- |
| 3D UNet       | 0.584       | 0.522       | 0.539       | 0.336       |
| 3D VNet       | 0.621       | 0.627       | 0.625       | 0.441       |
| 2.5D VNet     | 0.632       | 0.664       | 0.640       | 0.445       |


### Notes

This implementation differs from the VNet paper in a few ways:

**2.5D convolution**: We use the symmetric 5x5x5 convolution at the bottleneck of the encoder-decoder structure, and use 1x5x5 convolution in other layers.
