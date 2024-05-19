## Neural Audio Fingerprint implementation

This folder contains the code for reproducing [Neural Audio Fingerprint (NeuralFP)](https://arxiv.org/abs/2010.11910) paper.

This implementation is mostly inspired by [Original TensorFlow implementation](https://github.com/mimbres/neural-audio-fp) and [PyTorch implementation by Yi-Feng Chen](https://github.com/stdio2016/pfann).

[Pytorch lightning](https://lightning.ai/) library is used for fast model experimenting and configuration. The training script is `train_neuralfp.py` and the main training module is in `neuralfp/module/audio_fingerprint.py`.

## How to reproduce

Please visit this [Google Colab Notebook](https://colab.research.google.com/drive/1rVrzvJ7j-i2oMLl7F6zVcIZ_m79SEePA?usp=sharing) or [Kaggle Kernel](https://www.kaggle.com/huy1711/neural-audiofp-train) for installation, dataset preparation, and training script. Default training config is for GPU-enabled, if you do not have GPU, please change the config file at `configs/train.yaml`.

For local machine run, please follow the steps below.

### Dependencies
```bash
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Datasets

`fma_medium`
