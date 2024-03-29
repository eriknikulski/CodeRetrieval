import os
import torch

MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

SLURM_JOB_ID = os.getenv('SLURM_JOBID', 'local')

COMET_PROJECT_NAME = 'seq2seqtranslation'
COMET_WORKSPACE = 'eriknikulski'
COMET_EXP_NAME_LENGTH = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
SYNTH_PATH = 'data/synthetic/'

EVAL_PATH = PROJECT_PATH + 'eval/'
RELEVANCE_ANNOTATIONS_CSV_PATH = EVAL_PATH + 'annotationStore.csv'
MODEL_PREDICTIONS_CSV = EVAL_PATH + 'prediction.csv'
QUERY_CSV_PATH = EVAL_PATH + 'queries.csv'

DATA_PATH = SAVE_PATH + 'data/'
DATA_TRAIN_PATH = DATA_PATH + 'train_data.pickle'
DATA_TEST_PATH = DATA_PATH + 'test_data.pickle'
DATA_VALID_PATH = DATA_PATH + 'valid_data.pickle'
DATA_ALL_DF_PATH = DATA_PATH + 'all_data_df.pickle'
DATA_LANG_PATH = DATA_PATH + 'lang.pickle'
DATA_WORKING_PATH = DATA_PATH + 'working/'
DATA_WORKING_TRAIN_PATH = DATA_WORKING_PATH + 'train_data.pickle'
DATA_WORKING_TEST_PATH = DATA_WORKING_PATH + 'test_data.pickle'
DATA_WORKING_VALID_PATH = DATA_WORKING_PATH + 'valid_data.pickle'
DATA_FAIRSEQ_BASE_PATH = DATA_PATH + 'fairseq.'

PREPROCESS_BPE_PATH = SAVE_PATH + 'bpe/'
PREPROCESS_BPE_TRAIN_PATH = PREPROCESS_BPE_PATH + 'train.txt'
PREPROCESS_BPE_CODES_PATH = PREPROCESS_BPE_PATH + 'codes.txt'
PREPROCESS_BPE_VOCAB_PATH = PREPROCESS_BPE_PATH + 'vocab.txt'

MODEL_PATH = SAVE_PATH + 'model/'
MODEL_ENCODER_PATH = MODEL_PATH + SLURM_JOB_ID + '_encoder.pt'
MODEL_DECODER_PATH = MODEL_PATH + SLURM_JOB_ID + '_decoder.pt'
MODEL_JOINT_TRANSLATOR_PATH = MODEL_PATH + SLURM_JOB_ID + '_joint_translator.pt'
CHECKPOINT_PATH = MODEL_PATH + 'checkpoint/'

ANALYZE_PATH = SAVE_PATH + 'analyze/'
ANALYZE_VOCAB_HISTOGRAM = ANALYZE_PATH + 'vocab_hist.png'
ANALYZE_DATA_HISTOGRAM = ANALYZE_PATH + 'data_hist.png'
ANALYZE_OCCURRENCE = ANALYZE_PATH + 'occurrence/'

PROFILER_PATH = SAVE_PATH + 'profiler/'
PROFILER_TRACE_PATH = PROFILER_PATH + 'trace/'
PROFILER_STACKS_PATH = PROFILER_PATH + 'stacks/'

RAY_TUNE_LOCAL_DIR = SAVE_PATH + 'ray-tune/'

PROFILER_IS_ACTIVE = False
PROFILER_WAIT = 1
PROFILER_WARMUP = 1
PROFILER_ACTIVE = 1
PROFILER_REPEAT = 1

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
OOV_TOKEN = 3
NUMBER_TOKEN = '[number]'
TEXT_TOKEN = '[text]'
CHAR_TOKEN = '[char]'
PREPROCESS_VOCAB_SIZE = 20032
PREPROCESS_VOCAB_SIZE_DOC = 20032
PREPROCESS_VOCAB_SIZE_CODE = 10048
PREPROCESS_VOCAB_FREQ_THRESHOLD = 5
PREPROCESS_USE_BPE = False

IGNORE_PADDING_IN_LOSS = True
SHUFFLE_DATA = True
CUDA_DEVICE_COUNT = 0
NUM_WORKERS_DATALOADER = 0
PIN_MEMORY = True
MIN_CHECKPOINT_EPOCH = 500
MIN_CHECKPOINT_EPOCH_DIST = 100
LOG_BATCHES = True
LOG_IN_TRAINING = True
FP16 = True
ALLOW_TF32 = True
SET_GRADIENTS_NONE = True
DDP_FIND_UNUSED_PARAMETER = True

MIN_NUM_TOKENS = 5
MIN_LENGTH_DOCSTRING = 3
MAX_LENGTH_DOCSTRING = 25
MIN_LENGTH_CODE = 20
MAX_LENGTH_CODE = 100
CUT_LENGTHS = False

SIMPLE_CODE_ENCODER = False
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
BIDIRECTIONAL = 2
LSTM_ENCODER_DROPOUT = 0.3
LSTM_DECODER_DROPOUT = 0.3
HIDDEN_SIZE = 256
DROPOUT = 0.25

LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 4000
BATCH_SIZE = 256
BATCH_SIZE_VALID = 256
BATCH_SIZE_TEST = 256
LR_STEP_SIZE = 2000
LR_GAMMA = 0.1


def get_hyperparams(params=None):
    params = {} if not params else params
    return {
        'slurm_job_id': SLURM_JOB_ID,
        'profiler   is_active': PROFILER_IS_ACTIVE,
        'profiler   wait': PROFILER_WAIT,
        'profiler   warmup': PROFILER_WARMUP,
        'profiler   active': PROFILER_ACTIVE,
        'profiler   repeat': PROFILER_REPEAT,

        'setup   ignore_padding_in_loss': IGNORE_PADDING_IN_LOSS,
        'setup   shuffle_data': SHUFFLE_DATA,
        'setup   cuda_device_count': CUDA_DEVICE_COUNT,
        'setup   num_workers_dataloader': NUM_WORKERS_DATALOADER,
        'setup   pin_memory': PIN_MEMORY,
        'setup   min_checkpoint_epoch': MIN_CHECKPOINT_EPOCH,
        'setup   min_checkpoint_epoch_dist': MIN_CHECKPOINT_EPOCH_DIST,
        'setup   log_batches': LOG_BATCHES,
        'setup   log_in_training': LOG_IN_TRAINING,
        'setup   fp16': FP16,
        'setup   allow_tf32': ALLOW_TF32,
        'setup   set_gradients_none': SET_GRADIENTS_NONE,
        'setup   ddp_find_unused_parameter': DDP_FIND_UNUSED_PARAMETER,

        'data   min_num_tokens': MIN_NUM_TOKENS,
        'data   min_length_docstring': MIN_LENGTH_DOCSTRING,
        'data   max_length_docstring': MAX_LENGTH_DOCSTRING,
        'data   min_length_code': MIN_LENGTH_CODE,
        'data   max_length_code': MAX_LENGTH_CODE,
        'data   cut_lengths': CUT_LENGTHS,

        'model   simple_code_encoder': SIMPLE_CODE_ENCODER,
        'model   encoder_layers': ENCODER_LAYERS,
        'model   decoder_layers': DECODER_LAYERS,
        'model   bidirectional': BIDIRECTIONAL,
        'model   lstm_encoder_dropout': LSTM_ENCODER_DROPOUT,
        'model   lstm_decoder_dropout': LSTM_DECODER_DROPOUT,
        'model   hidden_size': HIDDEN_SIZE,
        'model   dropout': DROPOUT,

        'training   learning_rate': LEARNING_RATE,
        'training   momentum': MOMENTUM,
        'training   epochs': EPOCHS,
        'training   batch_size': BATCH_SIZE,
        'training   batch_size_valid': BATCH_SIZE_VALID,
        'training   batch_size_test': BATCH_SIZE_TEST,
        'training   lr_step_size': LR_STEP_SIZE,
        'training   lr_gamma': LR_GAMMA,
    } | params
