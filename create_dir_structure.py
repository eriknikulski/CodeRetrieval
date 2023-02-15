import pathlib

import const


def create_dirs():
    print('Creating directory structure...')
    paths = [
        const.PROJECT_PATH + 'data/java/',
        const.PROJECT_PATH + 'data/synthetic/train/',
        const.PROJECT_PATH + 'data/synthetic/test/',
        const.PROJECT_PATH + 'data/synthetic/valid/',
        const.EVAL_PATH,
        const.DATA_WORKING_PATH,
        const.PREPROCESS_BPE_TRAIN_PATH,
        const.PREPROCESS_BPE_CODES_PATH,
        const.PREPROCESS_BPE_VOCAB_PATH,
        const.MODEL_PATH,
        const.CHECKPOINT_PATH,
        const.ANALYZE_PATH,
        const.ANALYZE_OCCURRENCE,
        const.PROFILER_TRACE_PATH,
        const.PROFILER_STACKS_PATH,
    ]

    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    print('Finished.')


if __name__ == '__main__':
    create_dirs()
