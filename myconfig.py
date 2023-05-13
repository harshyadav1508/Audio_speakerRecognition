import multiprocessing
import datetime
import os

import torch

TRAIN_DATA_DIR = os.path.join(os.path.expanduser('~'), r"C:\Users\Harsh Yadav\PycharmProjects\dataset\LibriSpeech\train-clean-100")
TEST_DATA_DIR = os.path.join(os.path.expanduser('~'), r"C:\Users\Harsh Yadav\PycharmProjects\dataset\LibriSpeech-test\test-clean")

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filename = f"saved_model_{timestamp}.pt"
SAVED_MODEL_PATH = os.path.join(os.path.expanduser('~'), r"C:\Users\Harsh Yadav\PycharmProjects\SpeakerRecognitionFromScratch\saved_model",filename)

BATCH_SIZE = 8
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

TRAINING_STEPS = 100

USE_TRANSFORMER = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_MFCC = 40
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 3
BI_LSTM = True
FRAME_AGGREGATION_MEAN = True
LEARNING_RATE = 0.0001
SEQ_LEN = 100
SPECAUG_TRAINING = False