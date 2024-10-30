# Weight Copy and Low-Rank Adaptation for Few-Shot Distillation of Vision Transformers
### [Paper](https://arxiv.org/abs/2404.09326) | [Webpage]()
This repo holds the implementation of our paper *Weight Copy and Low-Rank Adaptation for Few-Shot Distillation of Vision Transformers* published at WACV 2025.
![](figures/fig_main.png)


## How to?

Clone repo, create a conda environment and install the dependencies:
```
git clone https://github.com/dianagrigore/WeCoLoRA
cd WeCoLoRA
conda create -n WeCoLoRA python=3.8
conda activate WeCoLoRA
pip install -r requirements.txt
```

## Weight Copy and Knowledge Distillation
To obtain the distilled student using our few-shot method:
```
python main_standard_kd.py \
--batch_size 128 --accum_iter 8 \
--epochs 10 --warmup_epochs 2 \
--teacher_model vit_base_patch16_224 \
--blr 1e-3 --weight_decay 0.05 \
--output_dir supervised_lora_kd_rank128_2_with_2percent \
--lora_distillation --lora_matrix_rank 128  \
--reduction_factor 2 --few_shot --few_shot_ratio=0.02
```

## Linear Probing Lora

```
python main_linprobe.py  --batch_size 512 --epochs 50 \
 --dataset=cifar-100 --blr 0.1 \
  --finetune supervised_lora_kd_rank128_2_with_2percent/checkpoint-9.pth  \
 --output_dir linprob_cifar_supervised_lora_kd_rank128_2_with_2percent \
 --model vit_base_patch16  --nb_classes 100 \
 --data_path=../../datasets/cifar \
 --reduction_factor 2 --lora_model --lora_matrix_rank=128
```


## How to Cite
```bibtex
@article{grigore2024weight,
  title={Weight Copy and Low-Rank Adaptation for Few-Shot Distillation of Vision Transformers},
  author={Grigore, Diana-Nicoleta and Georgescu, Mariana-Iuliana and Justo, Jon Alvarez and Johansen, Tor and Ionescu, Andreea Iuliana and Ionescu, Radu Tudor},
  journal={arXiv preprint arXiv:2404.09326},
  year={2024}
}
```
