import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = ROOT_DIR + '/'
SAVE_PATH = PROJECT_PATH + 'save/'
JAVA_PATH = 'data/java/java/final/jsonl/'
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
HIDDEN_SIZE = 256
TEACHER_FORCING_RATIO = 0.5
ITERS = 1000
