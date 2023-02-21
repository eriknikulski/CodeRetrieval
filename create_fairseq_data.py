import argparse

import const
import loader


parser = argparse.ArgumentParser(description='Creates dataset necessary for fairseq-preprocess')
parser.add_argument('-d', '--data', choices=['doc', 'code'], help='The data to be used.')

def create_fairseq_data(input_file, output_file, item='code_sequence'):
    data = loader.CodeDataset(input_file, to_tensors=False)
    data.enforce_length_constraints()
    with open(output_file, 'w') as fairseq_file:
        fairseq_file.writelines([f'{" ".join(elem)}\n' for elem in data.df[item]])


if __name__ == '__main__':
    args = parser.parse_args()
    if args.data == 'doc':
        train_path = const.DATA_DOC_TRAIN_FAIRSEQ_PATH
        valid_path = const.DATA_DOC_VALID_FAIRSEQ_PATH
        test_path = const.DATA_DOC_TEST_FAIRSEQ_PATH
    else:
        train_path = const.DATA_CODE_TRAIN_FAIRSEQ_PATH
        valid_path = const.DATA_CODE_VALID_FAIRSEQ_PATH
        test_path = const.DATA_CODE_TEST_FAIRSEQ_PATH

    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'train/', train_path)
    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'valid/', valid_path)
    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'test/', test_path)
