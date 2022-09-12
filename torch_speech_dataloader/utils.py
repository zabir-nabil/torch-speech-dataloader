import matplotlib.pyplot as plt
import torch
import numpy
import numpy as np
import random
"""
src: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#audio-i-o
"""

def save_waveform(waveform, sample_rate, filename = "wave.png", title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = 1, waveform.shape[0] # always assume single channel
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform, linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.savefig(filename)

def save_specgram(waveform, sample_rate, filename = "wave_spectrogram.png", title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.savefig(filename)

def crop_audio(y, audio_len_secs, sampling_rate, random_padding_if_short = False):
  if y.shape[0] < (audio_len_secs * sampling_rate):
      pad_req = (audio_len_secs * sampling_rate) - y.shape[0]
      if random_padding_if_short:
        pad_req_1 = random.randint(0, pad_req)
        pad_req_2 = pad_req - pad_req_1
        y = torch.nn.functional.pad(y, (int(pad_req_1), int(pad_req_2)))
      else:
        if pad_req % 2 == 0:
            y = torch.nn.functional.pad(y, (int(pad_req // 2), int(pad_req // 2)))
        else:
            y = torch.nn.functional.pad(y, (int(pad_req // 2), (int(pad_req // 2) + 1)))
  else:
      start_frame = np.int64(random.random() * (y.shape[0] - (audio_len_secs * sampling_rate)))
      y = y[start_frame:start_frame + int(audio_len_secs * sampling_rate)]

  return y