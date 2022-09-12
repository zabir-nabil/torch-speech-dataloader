import torch
import torchaudio

from torch_speech_dataloader.utils import save_waveform, save_specgram, crop_audio

import os, random
import traceback

import numpy as np

from torch_speech_dataloader.effects import Reverberation, MUSANAugmentation

from torch_speech_dataloader.augmentation_utils import placeholder_musan_config, placeholder_gpu_augmentation
"""
after reading audio file (in wav) -> audio should have dimension (d,) [1-d vector][for multi-channel audio, it will be converted to single-channel]
after feature extraction -> feature dimension should be (d_f, f) [2-d]
"""

class TorchSpeechDataset(torch.utils.data.Dataset):
    """PySpeech data pipeline."""

    def __init__(self, filenames, speech_labels, sampling_rate = 16000, audio_augmentation = [], rirs_reverb = {}, musan_augmentation = {}, features = None, feature_augmentation = [], device = torch.device('cpu'), verbose = 0, sanity_check_path = None, sanity_check_samples = 0):
        """
        augmentation = [(aug_1, p = 0.2), ...]
        verbose = [-1: no summary 0: basic summary stats after processing / error / fix 1: everything]
        """
        # assuming no vad is required
        self.filenames = filenames
        self.speech_labels = speech_labels
        self.audio_augmentation = audio_augmentation
        self.sampling_rate = sampling_rate
        self.feature_augmentation = feature_augmentation
        self.features = features
        self.device = device
        self.verbose = verbose

        # reverb
        self.reverb_apply = rirs_reverb.get("apply", False)
        self.reverb_source_files_path = rirs_reverb.get("reverb_source_files_path", ["torch_speech_dataloader/audio_assets/Room067-00068.wav"])
        if self.reverb_apply == True:
            self.reverb_aug = Reverberation(self.reverb_source_files_path, self.sampling_rate)

        # musan
        self.musan_apply = musan_augmentation.get("apply", False)
        self.musan_config = musan_augmentation.get("musan_config", placeholder_musan_config)
        self.musan_mix_multiples_max_count = musan_augmentation.get("mix_multiples_max_count", 0)
        self.musan_max_len = musan_augmentation.get("musan_max_len", -1)
        if self.musan_apply == True:
            self.musan_aug = MUSANAugmentation(self.musan_config, self.musan_mix_multiples_max_count, self.musan_max_len, self.sampling_rate)

        # sanity check
        self.sanity_check_counter = 0
        self.sanity_check_path = sanity_check_path
        self.sanity_check_samples = sanity_check_samples

        self.audio_len_secs = 4.5

        # speaker / speech classes
        self.speech_classes = sorted(list(set(self.speech_labels)))


    def __len__(self):
        return len(self.speech_labels)

    def __getitem__(self, idx):
        def itemprocessor(idx):
            # load audio
            y, sr = torchaudio.load(self.filenames[idx])
            y = y.mean(dim=0) # multi-channel audio -> single-channel audio
            if self.verbose > 0:
                print(f"[data read] shape of {self.filenames[idx]} :: {y.shape}")
            if sr != self.sampling_rate:
                # resample
                resample = torchaudio.transforms.Resample(sr, self.sampling_rate)
                y = resample(y)
                if self.verbose > 0:
                    print(f"fixing sampling rate from {sr} to {self.sampling_rate}")

            # y.shape = (d,)

            # make each audio of same length, for easy batching
            y = crop_audio(y, self.audio_len_secs, self.sampling_rate)

            # reverb
            if self.reverb_apply == True:
                y = self.reverb_aug.apply_augmentation(y)

            # musan
            if self.musan_apply == True:
                y = self.musan_aug.apply_augmentation(y)


            if len(self.audio_augmentation) > 0:
                for aug in self.audio_augmentation:
                    y = aug(y)
                if self.verbose > 0:
                    print(f"[after aug.] shape of {self.filenames[idx]} :: {y.shape}")


            # feature extraction
            if self.features != None:
                # assuming single feature right now
                y = self.features(y)
                if self.verbose > 0:
                    print(f"[after feat.] shape of {self.filenames[idx]} :: {y.shape}")

            if len(self.feature_augmentation) > 0:
                for aug in self.feature_augmentation:
                    y = aug(y)
                if self.verbose > 0:
                    print(f"[after feat. aug.] shape of {self.filenames[idx]} :: {y.shape}")

            # speech label
            sp_lab = self.speech_classes.index(self.speech_labels[idx])

            if self.sanity_check_path != None and self.sanity_check_counter < self.sanity_check_samples:
                try:
                    if len(y.shape) == 1:
                        audio_path = os.path.join(self.sanity_check_path, f"sc_ds_{self.sanity_check_counter}_{self.filenames[idx].split('.')[0].replace(' ', '')}_audio.wav")
                        wave_path = os.path.join(self.sanity_check_path, f"sc_ds_{self.sanity_check_counter}_{self.filenames[idx].split('.')[0].replace(' ', '')}_wave.png")
                        y_temp = torch.tensor(y)
                        torchaudio.save(audio_path, y_temp.reshape(1, -1), self.sampling_rate)
                        save_waveform(y_temp, self.sampling_rate, wave_path)
                    elif len(y.shape) == 2:
                        pass
                        # not implemented
                    self.sanity_check_counter += 1
                except Exception as e:
                    if self.verbose >= 0:
                        print(f"sanity check error :: {e}")
                    if self.verbose > 0:
                        print(traceback.format_exc())
            return y, sp_lab
        
        status = -1
        fail_count = 0
        while status != 1:
            try:
                y, sp_lab = itemprocessor(idx)
                status = 1
            except Exception as e:
                if self.verbose >= 0:
                    print(f"load / proc. error :: {e}")
                    print(traceback.format_exc())
                    idx = random.randint(0, len(self.speech_labels) - 1)
                    fail_count += 1
                    if fail_count == 10:
                        raise Exception("10 consecutive fails in load / proc. exiting ...")

        return y, sp_lab
                
class TorchSpeechDataLoader:
    def __init__(self, dataset, batch_size, num_workers, device, **kwargs):
        self.tsdl = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers)
        self.device = device
        self.batch_size = batch_size

        self.sanity_check_path = kwargs.get('sanity_check_path', None)
        self.sanity_check_samples = kwargs.get('sanity_check_samples', 0)
        self.verbose = kwargs.get('verbose', 0)
        self.sampling_rate = kwargs.get('sampling_rate', 16_000)
        self.sanity_check_counter = 0
        

        self.apply_augmentation = kwargs.get("batch_audio_augmentation", None)

    def get_batch(self):
        for speech, targets in self.tsdl:
            speech = speech.to(self.device).reshape(speech.shape[0], 1, -1)
            targets = targets.to(self.device)
            if self.apply_augmentation != None:
                speech = self.apply_augmentation(speech, sample_rate=16000)

            if self.sanity_check_path != None and self.sanity_check_counter < self.sanity_check_samples:
                try:
                    # randomly select a sample out of the batch size
                    eff_batch_sz = speech.shape[0]
                    y = speech[random.randint(0, eff_batch_sz - 1), 0, :].cpu()
                    if len(y.shape) == 1:
                        audio_path = os.path.join(self.sanity_check_path, f"sc_dl_{self.sanity_check_counter}_audio.wav")
                        wave_path = os.path.join(self.sanity_check_path, f"sc_dl_{self.sanity_check_counter}_wave.png")
                        torchaudio.save(audio_path, y.reshape(1, -1), self.sampling_rate)
                        save_waveform(y, self.sampling_rate, wave_path)
                    elif len(y.shape) == 2:
                        pass
                        # not implemented
                    self.sanity_check_counter += 1
                except Exception as e:
                    if self.verbose >= 0:
                        print(f"sanity check error :: {e}")
                    if self.verbose > 0:
                        print(traceback.format_exc())

            yield speech, targets
        
def get_torch_speech_dataloader(filenames, speech_labels, sampling_rate = 16000, rirs_reverb = {}, musan_augmentation = {}, audio_augmentation = [], features = None, feature_augmentation = [], device = torch.device('cpu'), batch_size = 1, num_workers = 0, verbose = 0, **kwargs):
    sanity_check_path = kwargs.get("sanity_check_path", None)
    sanity_check_samples = kwargs.get("sanity_check_samples", 0)
    batch_audio_augmentation = kwargs.get("batch_audio_augmentation", None)

    additional_params = {'sanity_check_path': sanity_check_path, 'sanity_check_samples': sanity_check_samples, 'verbose': verbose, 'sampling_rate': sampling_rate, 'batch_audio_augmentation': batch_audio_augmentation}

    tsdl_dataset = TorchSpeechDataset(filenames, speech_labels, sampling_rate, audio_augmentation, rirs_reverb, musan_augmentation, features, feature_augmentation, device, verbose, sanity_check_path, sanity_check_samples)
    tsdl_dataloader = torch.utils.data.DataLoader(dataset = tsdl_dataset, batch_size = batch_size, num_workers = num_workers)
    return tsdl_dataloader


def get_torch_speech_dataloader_from_config(config):
    filenames = config.get("filenames", [])
    speech_labels = config.get("speech_labels", [])
    sampling_rate = config.get("sampling_rate", 16_000)
    audio_augmentation = config.get("audio_augmentation", [])
    features = config.get("features", None)
    feature_augmentation = config.get("feature_augmentation", [])
    device = config.get("device", torch.device('cpu'))
    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 0)
    verbose = config.get("verbose", 0)
    sanity_check_path = config.get("sanity_check_path", None)
    sanity_check_samples = config.get("sanity_check_samples", 0)

    reverb = config.get("rirs_reverb", {})
    musan_augmentation = config.get("musan_augmentation", {})
    batch_audio_augmentation = config.get("batch_audio_augmentation", None)

    additional_params = {'sanity_check_path': sanity_check_path, 'sanity_check_samples': sanity_check_samples, 'verbose': verbose, 'sampling_rate': sampling_rate, 'batch_audio_augmentation': batch_audio_augmentation}

    tsdl_dataset = TorchSpeechDataset(filenames, speech_labels, sampling_rate, audio_augmentation, reverb, musan_augmentation, features, feature_augmentation, device, verbose, sanity_check_path, sanity_check_samples)
    tsdl_dataloader = TorchSpeechDataLoader(tsdl_dataset, batch_size, num_workers, device, **additional_params)
    return tsdl_dataloader
