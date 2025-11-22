# [AAAI 2026] Coverage-Constrained Human-AI Cooperation with Multiple Experts (CL2DC) 

This repo is the official implementation of our paper [Coverage-Constrained Human-AI Cooperation with Multiple Experts (CL2DC)]
![image](https://github.com/zhengzhang37/CL2DC/blob/main/architecture.png)


# Citation

If you use this code/data for your research, please cite our paper [Coverage-Constrained Human-AI Cooperation with Multiple Experts (CL2DC)](https://arxiv.org/abs/2411.11976).

```bibtex
@article{zhang2024coverage,
  title={Coverage-Constrained Human-AI Cooperation with Multiple Experts},
  author={Zhang, Zheng and Nguyen, Cuong and Wells, Kevin and Do, Thanh-Toan and Rosewarne, David and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2411.11976},
  year={2024}
}
```
## Pretrained AI model

We use [Promix](https://github.com/Justherozen/ProMix) for CIFAR-100 and ResNet34 for other datasets.

## Consensus Labels

We generate consensus labels via Multi-rater Learning methods [CrowdLab](https://github.com/cleanlab/cleanlab). The input of CrowdLab is any numbers of expert predictions (one-hot) and AI prediction (probability).

## CL2DC model training

```
python "methods/cl2dc.py"
```

## Other L2D methods

We also provide the codes for other L2D methods with penalty constraint training: [DCE (ICML 2024)](https://openreview.net/forum?id=aiz79FxjaI), [A-SM (NIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/791d3337291b2c574545aeecfa75484c-Paper-Conference.pdf), [MultiL2D (AISTATS 2023)](https://proceedings.mlr.press/v206/verma23a.html), [RS (AISTATS 2023)](https://proceedings.mlr.press/v206/mozannar23a), and [LECODU (ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-72992-8_9).
