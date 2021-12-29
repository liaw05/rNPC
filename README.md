# Diagnosis of recurrence of nasopharyngeal carcinoma

## Introduction

This repository is for our paper '[MRI-based deep learning model for surveillance and contour of local recurrence for nasopharyngeal carcinoma patients from national to community hospitals](http://xxxxx.pdf)'.

## Usage

### Training:

#### Detection and classification

Run python main_ddp.py

#### Segmentation

Run python main_seg.py


### Inference:

#### Detection and classification

Run python inference_cls.py

#### Segmentation

Run python inference_seg.py

## Result

#### Result of classification (AUC)

| Network       | SYSUCC test A set | SYSUCC test B set | SYSUCC test C set | SYSUCC test D set | Guangzhou test set | Zhongshan test set | Zhuhai test set |
| ------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------------ | ------------------ | --------------- |
| det+cls       | 0.91              | 0.90              | 0.94              | 0.86              | 0.89               | 0.88               | 0.86            |

#### Comparative Experiment of segmentation (DSC)

| Network       | Val set     | Test1 set   | SYSUCC set  | External set |
| ------------- | ----------- | ----------- | ----------- | ------------ |
| 3D UNet       | 0.584       | 0.522       | 0.539       | 0.336        |
| 3D VNet       | 0.621       | 0.627       | 0.625       | 0.441        |
| 2.5D VNet     | 0.632       | 0.664       | 0.640       | 0.445        |

## Citation

If you use xxx in your research, please cite the paper:
    
    @inproceedings{xxx,
      title={MRI-based deep learning model for surveillance and contour of local recurrence for nasopharyngeal carcinoma patients from national to community hospitals},
      author={xxx},
      journal={xxx},
      year={2022}
    }
