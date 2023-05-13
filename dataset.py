import glob
import os
import random

"""
example of flac file: r"'C:\\Users\\Harsh Yadav\\PycharmProjects\\dataset\\LibriSpeech\\train-clean-100\\19\\198\\19-198-0000.flac"
"""
def get_librispeech_speaker_to_utterance(data_dir):
    speaker_to_utterance = dict()
    flac_file = glob.glob(os.path.join(data_dir, "*", "*", "*.flac"))

    for file in flac_file:
        speaker_id = file.split("\\")[-3]
        utterance_id = file.split("\\")[-1].split(".")[0]
        if speaker_id not in speaker_to_utterance:
            speaker_to_utterance[speaker_id] = []
        speaker_to_utterance[speaker_id].append(file)
    return speaker_to_utterance


def get_triplet(spk_to_utts):
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)

