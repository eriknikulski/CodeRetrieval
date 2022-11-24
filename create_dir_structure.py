import pathlib

import const


def create_dirs():
    print('Creating directory structure...')
    pathlib.Path(const.PROJECT_PATH + 'data/').mkdir(parents=True, exist_ok=True)

    pathlib.Path(const.EVAL_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(const.DATA_WORKING_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(const.PREPROCESS_BPE_TRAIN_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(const.PREPROCESS_BPE_CODES_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(const.PREPROCESS_BPE_VOCAB_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(const.MODEL_PATH).mkdir(parents=True, exist_ok=True)

    pathlib.Path(const.ANALYZE_PATH).mkdir(parents=True, exist_ok=True)
    print('Finished.')


if __name__ == '__main__':
    create_dirs()
