# Zoom-VQA: Patches, Frames and Clips Integration for Video Quality Assessment

Kai Zhao, Kun Yuan, Ming Sun and Xing Wen

:rocket: **Updates History:**
- somthing more…
- April. 17, 2023: We release the Zoom-VQA source code and the final checkpoints for the NTIRE2023 final test phase. 

This repository is the official PyTorch implementation of 'Zoom-VQA: Patches, Frames and Clips Integration for Video Quality Assessment'. We won **the runner-up prize** in the NTIRE2023 Quality Assessment of Video Enhancement Challenge. 

## Network Architecture
As given in the figure, the overall framework of Zoom-VQA consists of two branchs, IQA branch and VQA branch. By combining the independent prediction scores of the two branches, the final prediction score is attained in a late fusion fashion.

![framework.png](images/framework.png)

## Datasets
### Overview
Download and extract VDPVE videos with annotations from [URL](https://arxiv.org/abs/2303.09290). To reproduce our results, we expect the whole directory struture should be organized as follows:
```
project/
  -data/
    -[train/val/test]
    -[train/val/test]_extract_imgs_2fps
  -IQA/
    -checkpoints/
    -logs/
    ...
  -VQA/
    -checkpoints/
    -logs/
    ...
```

### Preparation: extract frames 
As the the IQA branch is trained and inferred on frames, the first step is to extract frames from videos. Thus, we provide a script for extracting frames from videos in Python, while it still relys on FFmpeg. As follows:

```
# 1) extract train subset, 839 videos
python extract.py -d data/train/ -o data/train_extract_imgs_2fps -n 2
# 2) extract validation subset, 119 videos
python extract.py -d data/validation/ -o data/val_extract_imgs_2fps -n 2
# 3) extract test subset, 253 videos
python extract.py -d data/test/ -o data/test_extract_imgs_2fps -n 2
```

## Usage
### 1) IQA Branch

#### Evaluate on test subset
Please verify the current path is ’project/IQA’, download the checkpoint and save it in ‘project/IQA/checkpoints/’. After that, run the following command and the inferred results will be saved in ‘project/results/iqa_pred.txt' file. 

```bash
# infer on the test - FINAL TEST subset(253 videos), val - online val subset(119 videos)
python infer.py --dataset [test/val]
```

#### Train & Reproduce
```bash
python train_DDP_single.py --arch cpnet_multi --dataset vdpve_2fps --batch_size 32 --epochs 30 --lr 4e-4 --optimizer ADAMW --weight_decay 1e-2 --drop_path 0.1 --loss_type smooth_l1 --pretrain --rsize 512 --csize 320 --cutmix 0. --backbone_lr_decay 0.2 --pretrain --dropout 0.5 --suffix lr4e-4_b32_dp05_dpath01_wd1e-2_blrdecay02_reproduce
```

### 2) VQA Branch

VQA pipeline are almostly borrowed from [FAST-VQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA), which is trained/inferred on videos directly. Before executing following commands, please make sure your current path is ‘project/VQA/‘

### 3.1 Infer
Download the checkpoint and save it in ‘project/VQA/checkpoints/’. After that, run the following command and the inferred results will be saved in ‘project/results/vqa_pred.txt' file. 

```bash
# test: FINAL TEST subset(253 videos), ol_val: online val subset(119 videos), off_val: my split validation, 111 videos
python vqa_mine.py --rsize 480 --backbone swin_tiny_grpb --patch_size 6 --dataset [test/ol_val/off_val]
```

### 3.2 Train & Reproduce
Specifically, VideoSwinTransformer consumes pretty much GPU memory. So you should assure your GPU memory is larger than 24GB (V100 32GB is better).
```bash
python split_train.py --epochs 30 --lr 1e-3 --b_lr_decay 0.1 --backbone swin_tiny_grpb --batch_size 16 --rsize 480 --patch_size 6 --suffix lr1e-3_fragments_r480c336_blr01_reproduce
```

### 3) Late Fusion of Two Branches
After obtaining 'iqa_preds.txt' and 'vqa_preds.txt', you can average these preditions and fetch the final preds. 

```
# Jump to the root level of this project 
cd ./project

#run the ensemble script, the results will be saved in ‘project/results/output.txt’ file
python ensemble.py
```



## Acknowledgment
Our codes partially borrowed from [FAST-VQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) and [timm](https://github.com/rwightman/pytorch-image-models). Thanks for the [SwinIR](https://github.com/JingyunLiang/SwinIR) and [MANIQA](https://github.com/IIGROUP/MANIQA) Readme.md. We re-edit our file based on them.

## Related Works & Citations 
The following papers are to be cited, if relevant papers are propsed. 

1. [CVPRW 2023] Zoom-VQA: Patches, Frames and Clips Integration for Video Quality Assessment

```bibtex
@article{zhao2023zoomvqa,
      title={Zoom-VQA: Patches, Frames and Clips Integration for Video Quality Assessment}, 
      author={Zhao, Kai and Yuan, Kun and Sun, Ming and Wen, Xing},
      journal={arXiv preprint arXiv:2304.06440},
      year={2023}
}
```

2. [CVPR 2023] Quality-aware Pre-trained Models for Blind Image Quality Assessment

```bibtex
@article{zhao2023quality,
  title={Quality-aware Pre-trained Models for Blind Image Quality Assessment},
  author={Zhao, Kai and Yuan, Kun and Sun, Ming and Li, Mading and Wen, Xing},
  journal={arXiv preprint arXiv:2303.00521},
  year={2023}
}
```


