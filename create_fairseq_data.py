import argparse

import const
import loader


parser = argparse.ArgumentParser(description='Creates dataset necessary for fairseq-preprocess')
parser.add_argument('-d', '--data', choices=['doc', 'code'], action='append', nargs='*', help='The data to be used.')
parser.add_argument('-dirty', '--dirty', action='store_true', help='Keep the original dataset ond don\'t do preprocessing')

def create_fairseq_data(data, output_file, item='code_sequence'):
    with open(output_file, 'w+') as fairseq_file:
        fairseq_file.writelines([f'{" ".join(elem)}\n' for elem in data.df[item]])


if __name__ == '__main__':
    args = parser.parse_args()
    dirty = args.dirty
    data = args.data[0]
    if len(data) < 1 or len(data) > 2:
        raise Exception('Data argument should contain one or two values!')

    columns = {'doc': 'docstring_tokens', 'code': 'code_sequence'}
    dirty_str = 'dirty.' if dirty else ''
    folder_path = const.DATA_FAIRSEQ_BASE_PATH + dirty_str + data[0] + '-' + data[-1] + '/'

    train_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'train/', to_tensors=False, dirty=dirty)
    valid_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'valid/', to_tensors=False, dirty=dirty)
    test_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/', to_tensors=False, dirty=dirty)
    if not dirty:
        train_data.enforce_length_constraints()
        valid_data.enforce_length_constraints()
        test_data.enforce_length_constraints()

    for elem in data:
        train_path = folder_path + 'train.' + elem
        valid_path = folder_path + 'valid.' + elem
        test_path = folder_path + 'test.' + elem

        create_fairseq_data(train_data, train_path, item=columns[elem])
        create_fairseq_data(valid_data, valid_path, item=columns[elem])
        create_fairseq_data(test_data, test_path, item=columns[elem])
