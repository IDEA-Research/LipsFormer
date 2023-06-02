# LipsFormer



By Xianbiao Qi, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang.

This repo is the official implementation of ["LipsFormer: Introducing Lipschitz Continuity to Vision Transformers"](https://openreview.net/pdf?id=cHf1DcCwcH3).


Initial commits:

1. Pretrained models on ImageNet-1K ([LipsFormer-Swin-T-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)).

## Introduction

**LipsFormer** introduces a Lipschitz continuous Transformer to pursue training stability both theoretically and empirically for Transformer-based models. In contrast to previous practical tricks that address training instability by learning rate warmup, layer normalization, attention formulation, and weight initialization, we show that Lipschitz continuity is a more essential property to ensure training stability. In LipsFormer, we replace unstable Transformer component modules with Lipschitz continuous counterparts:  CenterNorm instead of LayerNorm, spectral initialization instead of Xavier initialization, scaled cosine similarity attention instead of dot-product attention, and weighted residual shortcut. We prove that these introduced modules are Lipschitz continuous and derive an upper bound on the Lipschitz constant of LipsFormer. .


