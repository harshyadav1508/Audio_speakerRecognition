import random

import librosa
import numpy as np
import torch

import dataset
import soundfile as sf

import myconfig
import specaug


def extract_features(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=myconfig.N_MFCC)

    return features.transpose()


def get_triplet_features(spk_to_utts):
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (extract_features(anchor_utt),
            extract_features(pos_utt),
            extract_features(neg_utt))


def trim_features(features, apply_specaug):
    """Trim features to SEQ_LEN, if number of rows are greater than SEQ_LEN(100), then it extract 100 rows from random position"""
    full_length = features.shape[0]
    start = random.randint(0, full_length - myconfig.SEQ_LEN)
    trimmed_features = features[start: start + myconfig.SEQ_LEN, :]
    if apply_specaug:
        trimmed_features = specaug.apply_specaug(trimmed_features)
    return trimmed_features


class TrimmedTripletFeaturesFetcher:
    def __init__(self, speaker_to_utterance):
        self.speaker_to_utterance = speaker_to_utterance

    def __call__(self, _):
        """Get a triplet of trimmed anchor/pos/neg features."""
        anchor, pos, neg = get_triplet_features(self.speaker_to_utterance)
        while anchor.shape[0] < myconfig.SEQ_LEN or pos.shape[0] < myconfig.SEQ_LEN or neg.shape[0] < myconfig.SEQ_LEN:
            anchor, pos, neg = get_triplet_features(self.speaker_to_utterance)
        return np.stack([trim_features(anchor, myconfig.SPECAUG_TRAINING),
                         trim_features(pos, myconfig.SPECAUG_TRAINING),
                         trim_features(neg, myconfig.SPECAUG_TRAINING)])


def get_batched_triplet_input(speaker_to_utterance, batch_size, pool):
    fetcher = TrimmedTripletFeaturesFetcher(speaker_to_utterance)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    # input_arrays is a list 0f 8 element where each element is a numpy array of shape (3, 100, 40)
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()

    # batch_input is a torch tensor of shape (24, 100, 40)
    return batch_input


def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows