from torch_speech_dataloader import get_torch_speech_dataloader, get_torch_speech_dataloader_from_config
import torch
import torchaudio
from timeit import Timer

from torch_speech_dataloader.augmentation_utils import placeholder_gpu_augmentation

dummy_tsdl = get_torch_speech_dataloader(filenames = ["../test.wav", "../test.wav"], speech_labels = ["test", "test2"], features = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,  hop_length=160, f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80), batch_size = 16)

for d, l in dummy_tsdl:
    print(d.shape)
    print(l.shape)

config_1 = {
    "filenames" : ["../test.wav"] * 5 + ["../test_hindi.wav"] * 5,
    "speech_labels" : ["test"] * 5 + ["test2"] * 5,
    "batch_size" : 3,
    "num_workers" : 5,
    "device" : torch.device('cuda:1'),
    "sanity_check_path" : "../sanity_test",
    "sanity_check_samples" : 2,
    "batch_audio_augmentation": placeholder_gpu_augmentation,
    "rirs_reverb" : {"apply": True},
    "musan_augmentation" : {"apply": True, "mix_multiples_max_count": -1, "musan_max_len": 1.},
    "verbose" : 1
}

config_2 = {
    "filenames" : ["../test.wav"] * 2500 + ["../test_hindi.wav"] * 2500,
    "speech_labels" : ["test"] * 2500 + ["test2"] * 2500,
    "batch_size" : 500,
    "num_workers" : 5,
    "device" : torch.device('cpu'),
    "sanity_check_path" : None,
    "sanity_check_samples" : 0
}

def test_2_wgpu(config):
    dummy_tsdl = get_torch_speech_dataloader_from_config(config)

    for d, l in dummy_tsdl.get_batch():
        print(d.shape)
        print(l)

def test_2_wogpu(config):
    dummy_tsdl = get_torch_speech_dataloader_from_config(config)

    for d, l in dummy_tsdl.get_batch():
        pass


test_2_wgpu(config_1)
#t = Timer(lambda: test_2_wgpu(config_1))
#print(t.timeit(number=5))

#t = Timer(lambda: test_2_wogpu(config_2))
#print(t.timeit(number=5))