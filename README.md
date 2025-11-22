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
