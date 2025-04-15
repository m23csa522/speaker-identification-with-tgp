
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random

class WavDataset(Dataset):
    def __init__(self, wav_list, label_map, preprocessor, max_length):
        self.wav_list = wav_list
        self.label_map = label_map
        self.preprocessor = preprocessor
        self.max_length = max_length

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path = self.wav_list[idx]
        label = self.label_map[os.path.basename(os.path.dirname(os.path.dirname(wav_path)))]
        waveform, sr = torchaudio.load(wav_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform[0]  # Use mono channel
        return waveform, len(waveform), label


class SpeakerDataset:
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.train_dataset = self._load_split(config.data["train_list"])
        self.test_dataset = self._load_split(config.data["test_list"])
        self.collate_fn = self._collate_fn

    def _load_split(self, list_path):
        with open(list_path, "r") as f:
            wav_paths = [line.strip() for line in f]

        with open(self.config.data["id_list"], "r") as f:
            speaker_ids = [line.strip() for line in f]
        label_map = {spk_id: idx for idx, spk_id in enumerate(speaker_ids)}

        return WavDataset(wav_paths, label_map, self.preprocessor, self.config.data["input_length"])

    def _collate_fn(self, batch):
        waveforms, lengths, labels = zip(*batch)
        max_len = max(lengths)

        padded_waveforms = []
        for wav in waveforms:
            if len(wav) < max_len:
                padded = torch.nn.functional.pad(wav, (0, max_len - len(wav)))
            else:
                padded = wav[:max_len]
            padded_waveforms.append(padded)

        return torch.stack(padded_waveforms), torch.tensor(lengths), torch.tensor(labels)
