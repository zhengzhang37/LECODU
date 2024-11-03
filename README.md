# [ECCV 2024] Learning to Complement or Defer to Multiple Users (LECODU) 

This repo is the official implementation of our paper [Learning to Complement or Defer to Multiple Users (LECODU)]([https://link.springer.com/chapter/10.1007/978-3-031-72992-8_9]).
![image](https://github.com/zhengzhang37/LECODU/blob/main/LECODU.png)

# Citation

If you use this code/data for your research, please cite our paper [Learning to Complement or Defer to Multiple Users (LECODU)](https://arxiv.org/abs/2407.07003).

```bibtex
@article{zhang2024learning,
  title={Learning to Complement and to Defer to Multiple Users},
  author={Zhang, Zheng and Ai, Wenjie and Wells, Kevin and Rosewarne, David and Do, Thanh-Toan and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2407.07003},
  year={2024}
}
```

## Pretrained LNL AI models 

We use [InstanceGM](https://github.com/arpit2412/InstanceGM) for [multi-rater IDN](https://github.com/xiaoboxia/Part-dependent-label-noise) dataset, [Promix](https://github.com/Justherozen/ProMix) for [CIFAR-10N](https://github.com/UCSC-REAL/cifar-10-100n) and [CIFAR-10H](https://openaccess.thecvf.com/content_ICCV_2019/html/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.html) dataset, and [NSHE](https://github.com/bupt-ai-cz/HSA-NRL) for Chaoyang dataset.

## Consensus Labels

We generate consensus labels via Multi-rater Learning methods [CrowdLab](https://github.com/cleanlab/cleanlab). The input of CrowdLab is any numbers of expert predictions (one-hot) and AI prediction (probability).

## Train LECODU
