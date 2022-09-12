import torch.nn.functional as F
import torch
import torchaudio
import random
from scipy import signal
import numpy
from torch_speech_dataloader.augmentation_utils import placeholder_musan_config
from torch_speech_dataloader.utils import crop_audio
"""
reverberation
torch implementation
numpy / scipy implementation: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/dataLoader.py#L66
"""

class Reverberation:
    def __init__(self, reverb_source_files_path = ["torch_speech_dataloader/audio_assets/Room067-00068.wav"], sampling_rate = 16_000):
        self.reverb_source_files_path = reverb_source_files_path
        self.sampling_rate = sampling_rate
    def apply_augmentation(self, audio):
        """
        audio is a single channel tensor
        torch.nn.functional.conv1d is very slow for large kernel size
        """
        rir_file = random.choice(self.reverb_source_files_path)
        rir, sr = torchaudio.load(rir_file)
        rir = rir.mean(axis = 0)
        if sr != self.sampling_rate:
            # resample
            resample = torchaudio.transforms.Resample(sr, self.sampling_rate)
            rir = resample(rir)
        def add_rev(audio, noise):
            rir = noise
            rir = rir / torch.sqrt(torch.sum(rir**2))

            audio = audio.view(1, 1, -1)
            rir = rir.view(1, 1, -1)

            return signal.convolve(audio, rir, mode='full')[0, 0, : audio.shape[-1]] 
        return add_rev(audio, rir) 

class MUSANAugmentation:
    def __init__(self, musan_config = placeholder_musan_config, mix_multiples_max_count = 0, musan_max_len = -1,  sampling_rate = 16_000):
        """
        musan_config = {
            "music": ([list of music file paths], range_for_num_music_files_to_use, range_for_noise_snr),
            "speech": ([list of speech file paths], range_for_num_speech_files_to_use, range_for_noise_snr),
        }
        mix_multiples_max_count = multiple noise types should be mixed (music + noise + ..) and how many noise types should be mixed at most
        musan_max_len: <= 0: take the musan noise and crop it with equal length (same as audio);
                        > 0: take the musan noise and crop it with length p (0.2, musan_max_len) if musan_max_len <= audio_len;
        """
        self.musan_config = musan_config
        self.mix_multiples_max_count = mix_multiples_max_count
        self.musan_max_len = musan_max_len
        self.sampling_rate = sampling_rate

    def apply_augmentation(self, audio):
        if self.mix_multiples_max_count == 0:
            noise_type = random.sample(list(self.musan_config.keys()), 1)
        elif self.mix_multiples_max_count == -1: # all types
            noise_type = list(self.musan_config.keys())
        else:
            if self.mix_multiples_max_count <= len(list(self.musan_config.keys())):
                noise_type = random.sample(list(self.musan_config.keys()), self.mix_multiples_max_count)
            else:
                noise_type = list(self.musan_config.keys())

        clean_db_snr = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        noises = []
        for nt in noise_type:
            noise_file_count = random.randint(self.musan_config[nt][1][0], self.musan_config[nt][1][1])
            if noise_file_count <= len(self.musan_config[nt][0]):
                noise_files = random.sample(self.musan_config[nt][0], noise_file_count)
            else:
                noise_files = self.musan_config[nt][0]

            
            for nf in noise_files:
                noiseaudio, sr  = torchaudio.load(nf)
                noiseaudio = noiseaudio.mean(axis = 0)

                if sr != self.sampling_rate:
                    # resample
                    resample = torchaudio.transforms.Resample(sr, self.sampling_rate)
                    noiseaudio = resample(noiseaudio)

                if self.musan_max_len > 0:
                    noiseaudio = crop_audio(noiseaudio, random.uniform(0.2, self.musan_max_len), self.sampling_rate) # min musan length = 0.2 sec
                
                noise_len = audio.shape[-1] / self.sampling_rate # required

                noiseaudio = crop_audio(noiseaudio, noise_len, self.sampling_rate, True)

                noise_snr   = random.uniform(self.musan_config[nt][2][0],self.musan_config[nt][2][1])
                noiseaudio = noiseaudio.numpy()
                noise_db_snr = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
                noise_factor = numpy.sqrt(10 ** ((clean_db_snr - noise_db_snr - noise_snr) / 10))
                noises.append(noise_factor * noiseaudio)

        noisy_out = numpy.sum(numpy.array(noises).reshape(len(noises), -1), axis = 0).flatten() + audio

        return noisy_out
                





