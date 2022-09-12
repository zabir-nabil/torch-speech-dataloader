# torch_speech_dataloader
A ready-to-use pytorch dataloader for audio classification, speech classification, speaker recognition, etc. with in-GPU augmentations.

 * PyTorch speech dataloader with 5 (or less) lines of code. `get_torch_speech_dataloader_from_config(config)`
 * Batch augmentation in GPU, powered by [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
 * RIRs augmentation with any set of IR file(s) [*cpu*]
 * MUSAN-like augmentation with any set of source files. Customizable. [*cpu*]
 * Written in one night, may contain bugs!

# Install

```cmd
pip install -U git+https://github.com/zabir-nabil/torch-speech-dataloader.git@main
```

# Use

```python
from torch_speech_dataloader import get_torch_speech_dataloader, get_torch_speech_dataloader_from_config
from torch_speech_dataloader.augmentation_utils import placeholder_gpu_augmentation

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
    "verbose" : 0
}

dummy_tsdl = get_torch_speech_dataloader_from_config(config_1)
for d, l in dummy_tsdl.get_batch():
    print(d.shape)
    print(l)
```