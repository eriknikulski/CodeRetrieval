import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
LOSS_PLOT_PATH = 'loss_plots/'
RELEVANCE_ANNOTATIONS_CSV_PATH = PROJECT_PATH + 'eval/annotationStore.csv'
MODEL_PREDICTIONS_CSV = PROJECT_PATH + 'eval/prediction.csv'
QUERY_CSV_PATH = PROJECT_PATH + 'eval/queries.csv'
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 11
HIDDEN_SIZE = 256
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.01
ITERS = 5000
