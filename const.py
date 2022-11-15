import os
import torch

MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

COMET_PROJECT_NAME = 'seq2seqtranslation'
COMET_WORKSPACE = 'eriknikulski'
COMET_EXP_NAME_LENGTH = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
SYNTH_PATH = 'data/synthetic/'
RELEVANCE_ANNOTATIONS_CSV_PATH = PROJECT_PATH + 'eval/annotationStore.csv'
MODEL_PREDICTIONS_CSV = PROJECT_PATH + 'eval/prediction.csv'
QUERY_CSV_PATH = PROJECT_PATH + 'eval/queries.csv'
TRAIN_DATA_SAVE_PATH = SAVE_PATH + 'train_data.pickle'
TRAIN_DATA_WORKING_PATH = SAVE_PATH + 'working_train_data.pickle'
TEST_DATA_SAVE_PATH = SAVE_PATH + 'test_data.pickle'
TEST_DATA_WORKING_PATH = SAVE_PATH + 'working_test_data.pickle'
VALID_DATA_SAVE_PATH = SAVE_PATH + 'valid_data.pickle'
VALID_DATA_WORKING_PATH = SAVE_PATH + 'working_valid_data.pickle'
ALL_DATA_DF_SAVE_PATH = SAVE_PATH + 'all_data_df.pickle'
INPUT_LANG_SAVE_PATH = SAVE_PATH + 'input_lang.pickle'
OUTPUT_LANG_SAVE_PATH = SAVE_PATH + 'output_lang.pickle'
PREPROCESS_TRAIN_PATH_DOC = SAVE_PATH + 'train_doc.txt'
PREPROCESS_TRAIN_PATH_CODE = SAVE_PATH + 'train_code.txt'
PREPROCESS_CODES_PATH_DOC = SAVE_PATH + 'codes_doc.txt'
PREPROCESS_CODES_PATH_CODE = SAVE_PATH + 'codes_code.txt'
ENCODER_PATH = SAVE_PATH + 'encoder.pt'
DECODER_PATH = SAVE_PATH + 'decoder.pt'
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
NUMBER_TOKEN = '[number]'
LABELS_ONLY = False
PREPROCESS_VOCAB_SIZE_DOC = 20000
PREPROCESS_VOCAB_SIZE_CODE = 20000
CUDA_DEVICE_COUNT = 1
IGNORE_PADDING_IN_LOSS = False
NUM_WORKERS_DATALOADER = 0
SHUFFLE_DATA = False

MIN_NUM_TOKENS = 5
MIN_LENGTH_DOCSTRING = 3
MAX_LENGTH_DOCSTRING = 20
MIN_LENGTH_CODE = 20
MAX_LENGTH_CODE = 50
HIDDEN_SIZE = 256
BIDIRECTIONAL = 1
TEACHER_FORCING_RATIO = 0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 200
BATCH_SIZE = 64
BATCH_SIZE_TEST = 64
TRAINING_PER_BATCH_PRINT = 1000
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
LR_STEP_SIZE = 100
LR_GAMMA = 0.1
GRADIENT_CLIPPING_ENABLED = 1
GRADIENT_CLIPPING_NORM_TYPE = 2
GRADIENT_CLIPPING_MAX_NORM = 2.0

HYPER_PARAMS = {
    'encoder_layers': ENCODER_LAYERS,
    'decoder_layers': DECODER_LAYERS,
    'bidirectional': BIDIRECTIONAL,
    'min_num_tokens': MIN_NUM_TOKENS,
    'min_length_docstring': MIN_LENGTH_DOCSTRING,
    'max_length_docstring': MAX_LENGTH_DOCSTRING,
    'min_length_code': MIN_LENGTH_CODE,
    'max_length_code': MAX_LENGTH_CODE,
    'cuda_device_count': CUDA_DEVICE_COUNT,
    'hidden_size': HIDDEN_SIZE,
    'learning_rate': LEARNING_RATE,
    'momentum': MOMENTUM,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'batch_size_test': BATCH_SIZE_TEST,
    'lr_step_size': LR_STEP_SIZE,
    'lr_gamma': LR_GAMMA,
}
