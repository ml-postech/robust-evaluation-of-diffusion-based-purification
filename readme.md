# Robust Evaluation of Diffusion-Based Adversarial Purification
This repository is the official implementation of ["Robust Evaluation of Diffusion-Based Adversarial Purification"](https://arxiv.org/abs/2303.09051) accepted by ICCV 2023, oral presentation.

## Abstract
We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.

## Requirements





## Pretrained models and data
Before executing our code, ensure to download the necessary pretrained models and the ImageNet dataset, if required. For both CIFAR-10 and ImageNet, we use diffusion models identical to those in [DiffPure](https://github.com/NVlabs/DiffPure#requirements). For SVHN, the [DDPM architecture](https://arxiv.org/abs/2006.11239) is adapted, and we have trained the model accordingly. We also use pretrained classifiers, and for both CIFAR-10 and ImageNet, you don't need to download the models, seperately. For SVHN, we have trained a WideResNet-28-10, available from [this source](https://github.com/szagoruyko/wide-residual-networks). After dowloading the models and dataset, you must update their path in `path.py`.


## Usages
There are several arguments to configure both attacks and defenses. `attack_method` specifies the chosen attack. `n_iter` and `eot` determine the number of update iterations and EOT (Expectation Over Transformation) samples, respectively. In our evaluation, since the surrogate process in an attack might require different purification settings, seperate arguments are used for specifying these details. For defense, `def_max_timesteps`, `def_num_denoising_steps`, and `def_sampling_method` determine the number of forward steps, the number of denoising steps, and the sampling method, respectively. For attack, `att_max_timesteps`, `att_num_denoising_steps`, and `att_sampling_method` are used. To implement multi-step purification, these parameters can accept comma-seperated values. For detailed information and practical examples, refer to our paper and the shell scripts in the `scripts` directory.


## Cite
Please cite our paper if you use the model or this code in your own work:
```
@article{lee2023robust,
  title={Robust evaluation of diffusion-based adversarial purification},
  author={Lee, Minjong and Kim, Dongwoo},
  journal={arXiv preprint arXiv:2303.09051},
  year={2023}
}
```