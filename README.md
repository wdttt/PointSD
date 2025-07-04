# PointSD

# [ICCV 2025] Harnessing Text-to-Image Diffusion Models for Point Cloud

By Yiyang Chen, Shanshan Zhao, Lunhao Duan, Changxing Ding and Dacheng Tao

This is the official implementation of "Harnessing Text-to-Image Diffusion Models for Point Cloud"
[[arXiv]](https://github.com/wdttt/PointSD)

![img](./figure/PointSD.png)

## Requirements

```
# Quick Start
conda create -n pointsd python=3.9 -y
conda activate pointsd

# Install pytorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install required packages
pip install -r requirements.txt
```
```
# Install fine-tune requirements
cd ./Point-MAE
pip install -r requirements.txt

# Install the extensions
# Chamfer Distance
cd ./extensions/chamfer_dist
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Datasets
See [DATASET.md](./DATASET.md) for details.

## PointSD Models
| Task              | Dataset        | Config                                                               | Acc.       | Download                                                                                     |
|-------------------|----------------|----------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | [train_pointsd.sh](./train_pointsd.sh)                        | N.A.       | [Pre-train](https://github.com/wdttt/PointSD)           |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./Point-MAE/cfgs/finetune_scan_objbg.yaml)     | 95.18%     | [OBJ_BG](https://github.com/wdttt/PointSD)          |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./Point-MAE/cfgs/finetune_scan_objonly.yaml) | 93.63%     | [OBJ_ONLY](https://github.com/wdttt/PointSD)        |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./Point-MAE/cfgs/finetune_scan_hardest.yaml) | 90.08%     | [PB_T50_RS](https://github.com/wdttt/PointSD)       |
| Classification    | ModelNet40 | [finetune_modelnet.yaml](./Point-MAE/cfgs/finetune_modelnet.yaml)         | 93.7%      | [ModelNet40](https://github.com/wdttt/PointSD)     |

| Task              | Dataset    | Config                                   | 5w10s (%)  | 5w20s (%)  | 10w10s (%) | 10w20s (%) | Download                                                                                       |
|-------------------|------------|------------------------------------------|------------|------------|------------|------------|------------------------------------------------------------------------------------------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](./Point-MAE/cfgs/fewshot.yaml) | 97.7 ± 1.8 | 99.0 ± 0.9 | 93.8±3.6 | 95.9±2.6 | [FewShot](https://github.com/wdttt/PointSD) |

| Task              | Dataset      | Config                                      | Cls.mIoU  | Insta.mIoU  | Download                                                                                     |
|-------------------|--------------|------------------------------------------|------------|------------|------------|
| Segmentation | ShapeNetPart | [segmentation](./Point-MAE/segmentation) | 84.5  | 86.1 |[Segmentation](https://github.com/wdttt/PointSD)|

## Pre-training
To pre-train PointSD, you need to set `task_name`, `MODEL_NAME` and `DATASET_NAME` in `train_pointsd.sh`. For the first training stage, you need to set `run_stage` to `stage1` and then run:
```
bash train_pointsd.sh
```
For the second training stage, you need to set `run_stage` to `stage2` and set the root of the first stage checkpoint  to `stage1_ckpt` and then run:
```
bash train_pointsd.sh
```
You can use checkpoint-120000 or checkpoint-best for subsequent fine-tuning.


## Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
# Select one config from finetune_scan_objbg/objonly/hardest.yaml
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model> --seed $RANDOM


# Test with fine-tuned ckpt
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_scan_hardest.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model> --seed $RANDOM

# Test with fine-tuned ckpt
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>

```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9> --seed $RANDOM
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --gpu <gpu_id> --ckpts <path/to/pre-trained/model> \
--log_dir <log_dir> --learning_rate 0.0002 --epoch 300 \
--root <path/to/data> \
--seed $RANDOM
```

## Acknowledgements

This codebase is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [VPD](https://github.com/wl-zhao/VPD/tree/main), [IPAdapter](https://github.com/tencent-ailab/IP-Adapter), [ULIP](https://github.com/salesforce/ULIP).

## Reference