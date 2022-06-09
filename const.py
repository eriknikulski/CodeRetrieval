import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
SYNTH_PATH = 'data/synthetic/'
RELEVANCE_ANNOTATIONS_CSV_PATH = PROJECT_PATH + 'eval/annotationStore.csv'
MODEL_PREDICTIONS_CSV = PROJECT_PATH + 'eval/prediction.csv'
QUERY_CSV_PATH = PROJECT_PATH + 'eval/queries.csv'
ENCODER_PATH = SAVE_PATH + 'encoder.pt'
DECODER_PATH = SAVE_PATH + 'decoder.pt'
SOS_TOKEN = 0
EOS_TOKEN = 1
NUMBER_TOKEN = '[number]'

MIN_NUM_TOKENS = 5
MAX_LENGTH = 20
MIN_LENGTH = 3
HIDDEN_SIZE = 256
BIDIRECTIONAL = 2
TEACHER_FORCING_RATIO = 0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 10
BATCH_SIZE = 64
BATCH_SIZE_TEST = 64
TRAINING_PER_BATCH_PRINT = 1000
ENCODER_LAYERS = 2
DECODER_LAYERS = 1

HYPER_PARAMS = {
    'encoder_layers': ENCODER_LAYERS,
    'decoder_layers': DECODER_LAYERS,
    'min_num_tokens': MIN_NUM_TOKENS,
    'max_length': MAX_LENGTH,
    'min_length': MIN_LENGTH,
    'hidden_size': HIDDEN_SIZE,
    'learning_rate': LEARNING_RATE,
    'momentum': MOMENTUM,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'batch_size_test': BATCH_SIZE_TEST,
}
