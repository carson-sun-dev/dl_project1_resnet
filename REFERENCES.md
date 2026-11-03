# References and provenance

- He et al., *Deep Residual Learning for Image Recognition*: https://arxiv.org/abs/1512.03385
  Residual learning and convolutional residual blocks.
- Zagoruyko and Komodakis, *Wide Residual Networks*: https://arxiv.org/abs/1605.07146
  Motivation for width/depth experiments; the `wide` configuration here is a custom
  CIFAR ResNet, not a claim of reproducing an exact published WRN.
- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch AMP documentation: https://docs.pytorch.org/docs/stable/amp.html
- Course-provided Project 1 competition and dataset instructions.

The scaffold was developed with GPT/Codex assistance for implementation, tests,
and documentation. Team members should review, understand, and adapt the code.
No external pretrained checkpoint or training dataset is bundled or downloaded.
Public implementation code has not been copied into this repository. Record the
source and applicable license if third-party code is added later.

MixUp/CutMix, distillation, ensembling and quantization are not implemented in
this initial scaffold. They must not be reported as completed experiments.
