import multiprocessing
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import dataset
import feature_extraction
import myconfig

class BaseSpeakerEncoder(nn.Module):
    def _load_from(self, saved_model):
        var_dict=torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])

class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self,saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)
        if saved_model:
            self._load_from(saved_model)

    def _aggregate_frames(self, batch_output):
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]

    def forward(self, x):
        D = 2 if myconfig.BI_LSTM else 1
        h0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        c0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    if myconfig.USE_TRANSFORMER:
        return TransformerSpeakerEncoder(load_from).to(myconfig.DEVICE)
    else:
        return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)


def train_network(speaker_to_utterance, num_steps, saved_model="", pool=None):
    losses = []
    start_time = time.time()
    encoder = get_speaker_encoder()

    #Train
    optimizer = torch.optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        #build batch input
        batch = feature_extraction.get_batched_triplet_input(speaker_to_utterance, myconfig.BATCH_SIZE, pool)

    return losses


def run_training():
    print("Training data:", myconfig.TRAIN_DATA_DIR)
    speaker_to_utterance = dataset.get_librispeech_speaker_to_utterance(myconfig.TRAIN_DATA_DIR)

    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(speaker_to_utterance,
                               myconfig.TRAINING_STEPS,
                               myconfig.SAVED_MODEL_PATH,
                               pool)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()



if __name__ == '__main__':
    run_training()

