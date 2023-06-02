# LipsFormer


By Xianbiao Qi, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang.

This repo is the official implementation of ["LipsFormer: Introducing Lipschitz Continuity to Vision Transformers"](https://openreview.net/pdf?id=cHf1DcCwcH3).


Initial commits:

1. Pretrained models on ImageNet-1K ([LipsFormer-Swin-T-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)).

In our paper, we compile two versions of LipsFormer. One is built on Swin and the other one is based on CSwin. In this repo, we release the code base based on Swin. You can easily change it to CSwin code.





## Introduction

**LipsFormer** introduces a Lipschitz continuous Transformer to pursue training stability both theoretically and empirically for Transformer-based models. In contrast to previous practical tricks that address training instability by learning rate warmup, layer normalization, attention formulation, and weight initialization, we show that Lipschitz continuity is a more essential property to ensure training stability. In LipsFormer, we replace unstable Transformer component modules with Lipschitz continuous counterparts:  CenterNorm instead of LayerNorm, spectral initialization instead of Xavier initialization, scaled cosine similarity attention instead of dot-product attention, and weighted residual shortcut. We prove that these introduced modules are Lipschitz continuous and derive an upper bound on the Lipschitz constant of LipsFormer.







## References
1. Qi, Xianbiao, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang. "LipsFormer: Introducing Lipschitz Continuity to Vision Transformers." In The Eleventh International Conference on Learning Representations. 2023.

2. Liu, Ze, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. "Swin transformer: Hierarchical vision transformer using shifted windows." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012-10022. 2021.

3. Dong, Xiaoyi, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. "Cswin transformer: A general vision transformer backbone with cross-shaped windows." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12124-12134. 2022.




