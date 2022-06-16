# OpenCoS: Contrastive Semi-supervised Learning for Handling Open-set Unlabeled Data
The code is compatible with CUDA 10.1 and python 3.6. See requirements.txt for all prerequisites, and you can also install them using the following command.
```
pip install -r requirements.txt
```

## Overview
* Stage 1. Unsupervised pre-training (--nproc_per_node=8, --ngpu 8: number of gpus; --dataset: cifar-10, cifar-100; --ooc_data: None, svhn, tiny; --model: ResNet50)
* Stage 2. OpenCoS + ReMixMatch (--sgpu: gpu id; --dataset: animal, cifar10, cifar100; --pc: 4, 25; --udata: svhn, tiny; --model: CIFAR_ResNet50_AuxBN; --model_path: pretrained simclr model)

  - --ood_samples: proportion of ooc (we use out-of-class 40,000 samples, in-class 10,000 samples)
  - --model_path: pre-trained model directory of Stage 1. (default: code/SimCLR/checkpoint folder)
  - --dataroot: CIFAR-10, CIFAR-100, SVHN datasets directory (default: /data folder)
  - --tinyroot: TinyImageNet dataset directory (default: /data/tinyimagenet/tiny-imagenet-200 folder)

## Running scripts (OpenCoS + ReMixMatch)
### CIFAR-Animals + CIFAR-Others benchmark
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-10 --model ResNet50 --batch-size 128 --name c10_U0 --ngpu 8 --ooc_data None
python3 train_opencos_remixmatch.py --sgpu 0 --dataset animal   --ema --model CIFAR_ResNet50_AuxBN --name opencos_remixmatch_Uothers_4pc --udata cten --pc 4 --naug 1 --batch-size 64 -ft --num_iters 50000 --model_path ../SimCLR/checkpoint/ckpt.t7c10_U0contrastive_learning_ResNet50_cifar-10_0 --ood_samples 40000 --lr 0.03 --fix_optim --lmd_pre 0 --lmd_rot 0 --lmd_unif 0.5 --aux_divide --ths 2 --use_jitter --temp_s2 0.1 --top_ratio 0.1
```

### CIFAR-10 + SVHN benchmark
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-10 --model ResNet50 --batch-size 128 --name c10_Usvhn40000 --ngpu 8 --ooc_data svhn
python3 train_opencos_remixmatch.py --sgpu 0 --dataset cifar10  --ema --model CIFAR_ResNet50_AuxBN --name opencos_remixmatch_Usvhn_4pc   --udata svhn --pc 4 --naug 1 --batch-size 64 -ft --num_iters 50000 --model_path ../SimCLR/checkpoint/ckpt.t7c10_Usvhn40000contrastive_learning_ResNet50_cifar-10_0 --ood_samples 40000 --lr 0.03 --fix_optim --lmd_pre 0 --lmd_rot 0 --lmd_unif 0.5 --aux_divide --ths 2 --use_jitter --temp_s2 0.1 --top_ratio 0.1
```

### CIFAR-10 + TinyImageNet benchmark
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-10 --model ResNet50 --batch-size 128 --name c10_Utiny40000 --ngpu 8 --ooc_data tiny
python3 train_opencos_remixmatch.py --sgpu 0 --dataset cifar10  --ema --model CIFAR_ResNet50_AuxBN --name opencos_remixmatch_Utiny_4pc   --udata tiny --pc 4 --naug 1 --batch-size 64 -ft --num_iters 50000 --model_path ../SimCLR/checkpoint/ckpt.t7c10_Utiny40000contrastive_learning_ResNet50_cifar-10_0 --ood_samples 40000 --lr 0.03 --fix_optim --lmd_pre 0 --lmd_rot 0 --lmd_unif 0.5 --aux_divide --ths 2 --use_jitter --temp_s2 0.1 --top_ratio 0.1
```

### CIFAR-100 + SVHN benchmark
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Usvhn40000 --ngpu 8 --ooc_data svhn
python3 train_opencos_remixmatch.py --sgpu 0 --dataset cifar100 --ema --model CIFAR_ResNet50_AuxBN --name opencos_remixmatch_Usvhn_4pc   --udata svhn --pc 4 --naug 1 --batch-size 64 -ft --num_iters 50000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Usvhn40000contrastive_learning_ResNet50_cifar-100_0 --ood_samples 40000 --lr 0.03 --fix_optim --lmd_pre 0 --lmd_rot 0 --lmd_unif 0.5 --aux_divide --ths 2 --use_jitter --temp_s2 0.1 --top_ratio 0.1
```

### CIFAR-100 + TinyImageNet benchmark
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
python3 train_opencos_remixmatch.py --sgpu 0 --dataset cifar100 --ema --model CIFAR_ResNet50_AuxBN --name opencos_remixmatch_Utiny_4pc   --udata tiny --pc 4 --naug 1 --batch-size 64 -ft --num_iters 50000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_ResNet50_cifar-100_0 --ood_samples 40000 --lr 0.03 --fix_optim --lmd_pre 0 --lmd_rot 0 --lmd_unif 0.5 --aux_divide --ths 2 --use_jitter --temp_s2 0.1 --top_ratio 0.1
```

## Running scripts for baseline methods (CIFAR-100 + TinyImageNet benchmark)
### Pre-training (wide_resnet / CIFAR_ResNet50)
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model wide_resnet --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
```

### SimCLR-le (wide_resnet / CIFAR_ResNet50)
```
python3 train.py --sgpu 0 --dataset cifar100 --multinomial --model wide_resnet --name multinomial_4pc --udata tiny --pc 4 -ft --batch-size 128 --ood_samples 40000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_wide_resnet_cifar-100_0
python3 train.py --sgpu 0            --dataset cifar100 --multinomial --model CIFAR_ResNet50 --name multinomial_4pc   --udata tiny --pc 4 -ft --batch-size 128 --ood_samples 40000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_ResNet50_cifar-100_0
```

### SimCLR-ft (wide_resnet / CIFAR_ResNet50)
```
python3 train.py --sgpu 0 --dataset cifar100 --model wide_resnet --name finetune_4pc --udata tiny --pc 4 -ft --batch-size 128 --ood_samples 40000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_wide_resnet_cifar-100_0 --lr 0.03 --fix_optim --num_iters 50000
python3 train.py --sgpu 0            --dataset cifar100               --model CIFAR_ResNet50 --name finetune_4pc      --udata tiny --pc 4 -ft --batch-size 128 --ood_samples 40000 --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_ResNet50_cifar-100_0 --lr 0.03 --fix_optim --num_iters 50000
```

### ReMixMatch-ft (wide_resnet / CIFAR_ResNet50)
```
python3 train_remixmatch.py --sgpu 0 --dataset cifar100 --ema --model wide_resnet --name remixmatch_4pc --udata tiny --pc 4 --naug 1 --batch-size 64 --num_iters 50000 --ood_samples 40000 --lr 0.03 --fix_optim --use_jitter --no_rampup --lmd_pre 0 --lmd_rot 0 -ft --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_wide_resnet_cifar-100_0
python3 train_remixmatch.py --sgpu 0 --dataset cifar100 --ema         --model CIFAR_ResNet50 --name remixmatch_4pc    --udata tiny --pc 4 --naug 1 --batch-size 64 --num_iters 50000 --ood_samples 40000 --lr 0.03 --fix_optim --use_jitter --no_rampup --lmd_pre 0 --lmd_rot 0 -ft --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_ResNet50_cifar-100_0
```

### FixMatch-ft (wide_resnet / CIFAR_ResNet50)
```
python3 train_fixmatch.py --sgpu 0 --dataset cifar100 --ema --model wide_resnet --name fixmatch_4pc --udata tiny --pc 4 --mu 1 --batch-size 64 --num_iters 50000 --ood_samples 40000 --lr 0.03 --fix_optim --use_jitter -ft --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_wide_resnet_cifar-100_0
python3 train_fixmatch.py --sgpu 0   --dataset cifar100 --ema         --model CIFAR_ResNet50 --name fixmatch_4pc      --udata tiny --pc 4 --mu 1 --batch-size 64 --num_iters 50000 --ood_samples 40000 --lr 0.03 --fix_optim --use_jitter -ft --model_path ../SimCLR/checkpoint/ckpt.t7c100_Utiny40000contrastive_learning_ResNet50_cifar-100_0
```
