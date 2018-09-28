# Speech Command Recognition

## A Keras implementation of neural attention model for speech command recognition

This repository presents a recurrent attention model designed to identify keywords in short segments of audio. It has been tested using the Google Speech Command Datasets (v1 and v2).
For a complete description of the architecture, please refer to [our paper](https://arxiv.org/abs/1808.08929).

Our main contributions are:

- A small footprint model (201K trainable parameters) that outperforms convolutional architectures for speech command recognition (AKA keyword spotting);
- A SpeechGenerator.py script that enables loading WAV files saved in .npy format from disk (like a Keras image generator, but for audio files);
- Attention outputs that make the model explainable (i.e., it is possible to identify what part of the audio was important to reach a conclusion).

# Attention Model

# How to use this code

## Cloning this repository

## Using Colab

Google Colaboratory is an amazing tool for experimentation using a Jupyter Notebook environment with GPUs.

## Train with your own data

# Final Words

If you find this code useful, please cite our work:

```
@ARTICLE{2018arXiv180808929C,
   author = {{Coimbra de Andrade}, D. and {Leo}, S. and {Loesener Da Silva Viana}, M. and 
	{Bernkopf}, C.},
    title = "{A neural attention model for speech command recognition}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1808.08929},
 keywords = {Electrical Engineering and Systems Science - Audio and Speech Processing, Computer Science - Sound},
     year = 2018,
    month = aug,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180808929C},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
