import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
SYNTH_PATH = 'data/synthetic/'
LOSS_PLOT_PATH = 'loss_plots/'
RELEVANCE_ANNOTATIONS_CSV_PATH = PROJECT_PATH + 'eval/annotationStore.csv'
MODEL_PREDICTIONS_CSV = PROJECT_PATH + 'eval/prediction.csv'
QUERY_CSV_PATH = PROJECT_PATH + 'eval/queries.csv'
ENCODER_PATH = SAVE_PATH + 'encoder.pt'
DECODER_PATH = SAVE_PATH + 'decoder.pt'
SOS_token = 0
EOS_token = 1

MIN_NUM_TOKENS = 5
MAX_LENGTH = 16
HIDDEN_SIZE = 256
BIDIRECTIONAL = 2
TEACHER_FORCING_RATIO = 0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 100
BATCH_SIZE = 10
BATCH_SIZE_TEST = 10
TRAINING_PER_BATCH_PRINT = 1000

HYPER_PARAMS = {
    'min_num_tokens': MIN_NUM_TOKENS,
    'max_length': MAX_LENGTH,
    'hidden_size': HIDDEN_SIZE,
    'learning_rate': LEARNING_RATE,
    'momentum': MOMENTUM,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'batch_size_test': BATCH_SIZE_TEST,
}
