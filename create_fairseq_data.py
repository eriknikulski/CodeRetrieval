import const
import loader


def create_fairseq_data(input_file, output_file, item='code_sequence'):
    data = loader.CodeDataset(input_file, to_tensors=False)
    data.enforce_length_constraints()
    with open(output_file, 'w') as fairseq_file:
        fairseq_file.writelines([f'{" ".join(elem)}\n' for elem in data.df[item]])


if __name__ == '__main__':
    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'train/', const.DATA_TRAIN_FAIRSEQ_PATH)
    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'test/', const.DATA_TEST_FAIRSEQ_PATH)
    create_fairseq_data(const.PROJECT_PATH + const.JAVA_PATH + 'valid/', const.DATA_VALID_FAIRSEQ_PATH)
