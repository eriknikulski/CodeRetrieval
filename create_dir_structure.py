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
        const.DATA_FAIRSEQ_BASE_PATH + 'doc-doc',
        const.DATA_FAIRSEQ_BASE_PATH + 'doc-code',
        const.DATA_FAIRSEQ_BASE_PATH + 'code-doc',
        const.DATA_FAIRSEQ_BASE_PATH + 'code-code',
        const.DATA_PATH + 'with_url/',
        const.DATA_PATH + 'with_url/fairseq.doc-doc',
        const.DATA_PATH + 'with_url/fairseq.doc-code',
        const.DATA_PATH + 'with_url/fairseq.code-doc',
        const.DATA_PATH + 'with_url/fairseq.code-code',
        const.PREPROCESS_BPE_PATH,
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
