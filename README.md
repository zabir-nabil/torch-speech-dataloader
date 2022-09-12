## torch-speech-dataloader
A ready-to-use pytorch dataloader for audio classification, speech classification, speaker recognition, etc. with in-GPU augmentations.

 * PyTorch speech dataloader with 5 (or less) lines of code. `get_torch_speech_dataloader_from_config(config)`
 * Batch augmentation in GPU, powered by [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
 * RIRs augmentation with any set of IR file(s) [*cpu*]
 * MUSAN-like augmentation with any set of source files. Customizable. [*cpu*]
 * Written in one night, may contain bugs!

## Install

```cmd
pip install -U git+https://github.com/zabir-nabil/torch-speech-dataloader.git@main
```

## Use

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

## Others

#### `config` parameters

 * `filenames`: A list of filepaths for the audio / speech files (usually wav).
 * `speech_labels`: Corresponding labels for `filenames` / list of audio files.
 * `batch_size`: Batch size of the dataloader.
 * `num_workers`: Dataloader workers.
 * `device`: torch device [default: *cpu*].
 * `sanity_check_path`: If you want to look at the sample audio files generated, specify a path where the sample augmented audio files will be saved.
 * `sanity_check_samples`: Number of sample audio files to store in the sanity check folder.
 * `batch_audio_augmentation`: Usually, it will run on the GPU batch if gpu device is specified, else on the CPU batch. Any transform (compose) / augmentation, that takes a tensor of dimension **[B x C x N]**.
 * `rirs_reverb`:
   * `apply`: If apply is true, only then this augmentation will be applied to each audio individually.
   * `reverb_source_files_path`: A list of IR filepaths.
 * `musan_augmentation`:
   * `apply`: If apply is true, only then this augmentation will be applied to each audio individually.
   * `musan_config`: 
         ```{
            "music": ([list of music file paths], range_for_num_music_files_to_use, range_for_noise_snr),
            "speech": ([list of speech file paths], range_for_num_speech_files_to_use, range_for_noise_snr),
        }``` `[example: augmentation_utils.placeholder_musan_config]`
   * `mix_multiples_max_count`: Multiple noise types should be mixed (music + noise + `...`). Number of noise types that should be mixed at most.
   * `musan_max_len`: `<= 0`: take the musan noise and crop it with equal length (same as input audio); `> 0`: maximum length of the cropped musan noise (in secs.).
 * `audio_augmentation`: List of `func`s that can be applied to a single audio with shape **[N,]**. 
 * `features`: Feature extraction. **[N,]** -> **[T,F]**.
 * `feature_augmentation`: List of `func`s that can be applied to a single feature with shape **[T,F]**.