# References and attribution

These references identify methods actually implemented in the final model or
saved historical studies. Historical studies are not all included as executable
source in the final package. A citation does not claim reproduction of an entire
paper protocol or a competition improvement. Links and bibliographic details
were retained from the project's existing records; no new literature review was
performed for this release.

## Final model and training

- He, Zhang, Ren and Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR 2016. Residual-learning foundation.
- He, Zhang, Ren and Sun. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027). ECCV 2016. Preactivation block ordering.
- Zagoruyko and Komodakis. [Wide Residual Networks](https://arxiv.org/abs/1605.07146). BMVC 2016. Width scaling and WRN-22-4 topology; shortcut behavior was checked against the [authors' implementation](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua). This project uses its own initialization policy and zero dropout.
- Zhang, Cisse, Dauphin and Lopez-Paz. [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR 2018. Convex input/target mixing.
- DeVries and Taylor. [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552). 2017. Input masking regularization.
- Foret, Kleiner, Mobahi and Neyshabur. [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412). ICLR 2021. Two-pass optimization and the SAM comparison.
- Kwon, Kim, Park and Choi. [ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks](https://arxiv.org/abs/2102.11600). ICML 2021. Adaptive perturbation in the final recipe. SAM/ASAM equations were implemented independently; no copied research-optimizer implementation is claimed.
- PyTorch [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html), and [cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html); torchvision 0.23 [ColorJitter](https://docs.pytorch.org/vision/0.23/generated/torchvision.transforms.ColorJitter.html) and [RandomGrayscale](https://docs.pytorch.org/vision/0.23/generated/torchvision.transforms.RandomGrayscale.html). EMA is a project implementation with the recurrence and copied-buffer policy specified in [TRAINING.md](TRAINING.md), not a claim to implement SWA.

## Shake-Shake route

- Gastaldi. [Shake-Shake Regularization](https://arxiv.org/abs/1705.07485). 2017.
  The teammate implementation in `src/shake_shake.py` identifies an independent
  reimplementation of the [author code](https://github.com/xgastaldi/shake-shake),
  including spatially shifted projection paths. See the
  [teammate reproduction guide](../experiments/SHAKE899_REPRODUCTION.md).
  Upstream code is not a source of pretrained weights. Preserve required notices
  if incorporating upstream code.

## Other implemented studies

- Hendrycks et al. [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781). ICLR 2020. The operator/mixing code adapts [google-research/augmix](https://github.com/google-research/augmix), inspected revision `9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`. Preserve [third_party/augmix/LICENSE](../third_party/augmix/LICENSE), [NOTICE](../third_party/augmix/NOTICE), corresponding final-package notices and embedded source headers.
- Pinto, Yang, Lim, Torr and Dokania. [RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness](https://arxiv.org/abs/2206.14502). NeurIPS 2022; [author implementation](https://github.com/FrancescoPinto/RegMixup). Historical adapted joint unmixed/mixed objective; not an exact paper reproduction. Its separate frozen research material retains its MIT-reference-license provenance.
- Xie et al. [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431). CVPR 2017. ResNeXt study.
- Khosla et al. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). NeurIPS 2020. Joint two-view CE/SupCon study with a matched CE control; not the paper's two-stage protocol.
- Yu and Koltun. [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122). ICLR 2016. Hu, Shen and Sun. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507). CVPR 2018. Zhang. [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486). ICML 2019. Mechanisms used in the custom multiscale/SE/anti-aliasing studies.
- Hinton, Vinyals and Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 2015. Furlanello et al. [Born Again Neural Networks](https://arxiv.org/abs/1805.04770). ICML 2018. Historical same-capacity student study using a scratch-trained permitted-data teacher; it failed the promotion gate and is not deployed.
- Izmailov et al. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). UAI 2018. Motivation for a rejected fixed retrospective averaging probe, distinct from the paper's constant/cyclical-LR protocol and from final EMA. BN recalibration used [PyTorch 2.8 SWA utilities](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/swa_utils.py) under PyTorch's BSD-style license.
- Hendrycks and Dietterich. [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261). ICLR 2019. Motivation for controlled stress diagnostics on permitted validation images; no external CIFAR-10-C data or official mCE benchmark was used.

## Course provenance and assistance

The [course competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)
and user-provided official course instructions define the permitted data,
architecture constraints and delivery requirements. The public documentation does
not claim a fresh retrieval of the private course rubric or current selection
state.

GPT/Codex assisted implementation, debugging, experiment work and documentation.
Team members remain responsible for reviewing, understanding and presenting the
work accurately. All final-model learned weights originate from scratch training
on the authorized supplied training data; citations are not sources of
pretrained checkpoints. Preserve existing third-party licenses and notices.
No root software license is declared, so the project as a whole is not labeled
Apache-2.0 merely because the adapted AugMix component uses that license.
