from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, BandPassFilter, BandStopFilter, AddColoredNoise,  PeakNormalization, TimeInversion
import torch_speech_dataloader
import os

placeholder_musan_config = {
    "music": ([f"{os.path.dirname(torch_speech_dataloader.__file__)}/audio_assets/music_1.wav"], (1, 1), (0, 15)),
    "speech": ([f"{os.path.dirname(torch_speech_dataloader.__file__)}/audio_assets/speech_1.wav"], (1, 1), (13, 20))
}

placeholder_gpu_augmentation = Compose(
                                    transforms=[
                                        AddBackgroundNoise(
                                            [f"{os.path.dirname(torch_speech_dataloader.__file__)}/audio_assets/music_1.wav"],
                                            p = 0.3, sample_rate = 16_000
                                        ),
                                        Gain(
                                            min_gain_in_db=-15.0,
                                            max_gain_in_db=5.0,
                                            p=0.5,
                                        ),
                                        BandPassFilter(p = 0.3, sample_rate = 16_000),
                                        BandStopFilter(p = 0.3, sample_rate = 16_000),

                                        AddColoredNoise(p = 0.3, sample_rate = 16_000),
                                        PeakNormalization(apply_to = "only_too_loud_sounds", p = 0.2, sample_rate = 16_000),
                                        TimeInversion(p = 1.0, sample_rate = 16_000),
                                        PolarityInversion(p=0.5)
                                    ]
                                )