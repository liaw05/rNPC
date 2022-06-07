# Diagnosis of recurrence of nasopharyngeal carcinoma

## Introduction

<!-- This repository is for our paper '[MRI-based deep learning models for the detection of local recurrence in nasopharyngeal carcinoma](http://xxxxx.pdf)'. -->
This repository is for our paper 'MRI-based deep learning models for the detection of local recurrence in nasopharyngeal carcinoma'.

## Usage

### Training:

#### Detection and classification

Run python main_ddp.py

#### Segmentation (Using 2.5D VNet)

Run python main_seg.py


### Inference:

#### Detection and classification

Run python inference_cls.py

#### Segmentation (Using 2.5D VNet)

Run python inference_seg.py

## Result

#### Result of classification (AUC)

| Network       | SYSUCC test A set | SYSUCC test B set | SYSUCC test C set | SYSUCC test D set | Guangzhou test set | Zhongshan test set | Zhuhai test set |
| ------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------------ | ------------------ | --------------- |
| det+cls       | 0.905             | 0.913             | 0.946             | 0.871             | 0.897              | 0.870              | 0.844           |

#### Result of segmentation (DSC)

| Network       | Val set     | Test1 set   | SYSUCC set  | External set |
| ------------- | ----------- | ----------- | ----------- | ------------ |
| 3D UNet       | 0.584       | 0.522       | 0.539       | 0.336        |
| 3D VNet       | 0.621       | 0.627       | 0.625       | 0.441        |
| 2.5D VNet       | 0.632       | 0.664       | 0.640       | 0.445        |
| [nnUNet](https://github.com/MIC-DKFZ/nnUNet)        | 0.674       | 0.671       | 0.662       | 0.497        |

## Citation

If you found this code useful for your research, please cite the paper.
<!---
    
    @inproceedings{xxx,
      title={MRI-based deep learning models for the detection of local recurrence in nasopharyngeal carcinoma},
      author={xxx},
      journal={xxx},
      year={2022}
    }
-->
