# LipsFormer

By Xianbiao Qi, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang.

This repo is the official implementation of ["LipsFormer: Introducing Lipschitz Continuity to Vision Transformers"](https://openreview.net/pdf?id=cHf1DcCwcH3).

Initial commits:
We release one pretrained model below, the result of the model is 82.70 for LipsFormer-Swin-Tiny. We train the model without using any warmup.

1. Pretrained models on ImageNet-1K ([LipsFormer-Swin-T-IN1K](https://github.com/cyh1112/LipsFormer/releases/download/checkpoint/lipsformer-swin-tiny.pth)).

In our paper, we compile two versions of LipsFormer. One is built on Swin and the other one is based on CSwin, We do not merge these two code bases. Thank the authors of CSwin and Swin for releasing their code base.
In this repo, we release the code base based on Swin. You can easily change it to CSwin code. 

<br/>

## Introduction

**LipsFormer** introduces a Lipschitz continuous Transformer to pursue training stability both theoretically and empirically for Transformer-based models. In contrast to previous practical tricks that address training instability by learning rate warmup, layer normalization, attention formulation, and weight initialization, we show that Lipschitz continuity is a more essential property to ensure training stability. In LipsFormer, we replace unstable Transformer component modules with Lipschitz continuous counterparts:  CenterNorm instead of LayerNorm, spectral initialization instead of Xavier initialization, scaled cosine similarity attention instead of dot-product attention, and weighted residual shortcut. We prove that these introduced modules are Lipschitz continuous and derive an upper bound on the Lipschitz constant of LipsFormer.

<br/>

## Training
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 32 
```

<br/>

## References
```
@misc{qi2023lipsformer,
      title={LipsFormer: Introducing Lipschitz Continuity to Vision Transformers}, 
      author={Xianbiao Qi and Jianan Wang and Yihao Chen and Yukai Shi and Lei Zhang},
      year={2023},
      eprint={2304.09856},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{liu2021swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}, 
      author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
      year={2021},
      eprint={2103.14030},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{dong2022cswin,
      title={CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows}, 
      author={Xiaoyi Dong and Jianmin Bao and Dongdong Chen and Weiming Zhang and Nenghai Yu and Lu Yuan and Dong Chen and Baining Guo},
      year={2022},
      eprint={2107.00652},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


