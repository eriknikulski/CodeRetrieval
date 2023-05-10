import argparse
import pickle

from subword_nmt import subword_nmt

import const
import data
import loader

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False,
                    help='Do not remove duplicates in data.')


def run(args):
    if args.data == 'java':
        data_path = const.JAVA_PATH
    else:
        data_path = const.SYNTH_PATH

    remove_duplicates = not args.keep_duplicates

    train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/',
                                    build_language=False, remove_duplicates=remove_duplicates, to_tensors=False)
    test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/',
                                   build_language=False, remove_duplicates=remove_duplicates, to_tensors=False)
    valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/',
                                    build_language=False, remove_duplicates=remove_duplicates, to_tensors=False)

    print('Creating training files...')
    with open(const.PREPROCESS_BPE_TRAIN_PATH, 'w', encoding='utf-8') as train_file:
        for text in train_data.df['docstring_tokens']:
            train_file.write(f'{" ".join(text)}\n')
        for text in test_data.df['docstring_tokens']:
            train_file.write(f'{" ".join(text)}\n')
        for text in valid_data.df['docstring_tokens']:
            train_file.write(f'{" ".join(text)}\n')
        for text in train_data.df['code_sequence']:
            train_file.write(f'{" ".join(text)}\n')
        for text in test_data.df['code_sequence']:
            train_file.write(f'{" ".join(text)}\n')
        for text in valid_data.df['code_sequence']:
            train_file.write(f'{" ".join(text)}\n')

    if const.PREPROCESS_USE_BPE:
        print('Creating codes files...')
        with open(const.PREPROCESS_BPE_TRAIN_PATH, encoding='utf-8') as train_file, \
                open(const.PREPROCESS_BPE_CODES_PATH, 'w', encoding='utf-8') as codes_file:
            subword_nmt.learn_bpe(train_file, codes_file, const.PREPROCESS_VOCAB_SIZE_DOC)

        print('Creating vocab files...')
        with open(const.PREPROCESS_BPE_TRAIN_PATH, encoding='utf-8') as train_file, \
                open(const.PREPROCESS_BPE_VOCAB_PATH, 'w', encoding='utf-8') as vocab_file:
            subword_nmt.get_vocab(train_file, vocab_file)

        print('Applying codes...')
        with open(const.PREPROCESS_BPE_CODES_PATH, encoding='utf-8') as codes_file, \
                open(const.PREPROCESS_BPE_VOCAB_PATH, encoding='utf-8') as vocab_file:
            vocab = subword_nmt.read_vocabulary(vocab_file, const.PREPROCESS_VOCAB_FREQ_THRESHOLD)
            bpe = subword_nmt.BPE(codes_file, vocab=vocab)
            train_data.df[['docstring_tokens']] = train_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
            test_data.df[['docstring_tokens']] = test_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
            valid_data.df[['docstring_tokens']] = valid_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
            train_data.df[['code_sequence']] = train_data.df[['code_sequence']].applymap(bpe.segment_tokens)
            test_data.df[['code_sequence']] = test_data.df[['code_sequence']].applymap(bpe.segment_tokens)
            valid_data.df[['code_sequence']] = valid_data.df[['code_sequence']].applymap(bpe.segment_tokens)

    print('Building languages...')
    lang = data.Lang('lang')
    train_data.build_language(language=lang)
    test_data.build_language(language=lang)
    valid_data.build_language(language=lang)

    print('write csv files...')
    with open(const.DATA_PATH + 'code_sequences.csv', 'w') as f:
        f.write(train_data.df[['code_sequence']].applymap(' '.join).to_csv())
    with open(const.DATA_PATH + 'code_tokens.csv', 'w') as f:
        f.write(train_data.df[['code_tokens']].applymap(' '.join).to_csv())
    with open(const.DATA_PATH + 'methode_name.csv', 'w') as f:
        f.write(train_data.df[['methode_name']].applymap(' '.join).to_csv())

    print('Converting to tensors...')
    train_data.to_tensors()
    test_data.to_tensors()
    valid_data.to_tensors()

    print('Saving...')
    pickle.dump(train_data, open(const.DATA_TRAIN_PATH, 'wb'))
    pickle.dump(test_data, open(const.DATA_TEST_PATH, 'wb'))
    pickle.dump(valid_data, open(const.DATA_VALID_PATH, 'wb'))
    pickle.dump(lang, open(const.DATA_LANG_PATH, 'wb'))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
