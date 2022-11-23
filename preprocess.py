import argparse
import pickle

import pandas as pd
from subword_nmt import subword_nmt

import const
import data
import loader

parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')
parser.add_argument('-lo', '--labels-only', action='store_true', default=False, help='The data to be used.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False, help='Do not remove duplicates in data.')


def run(args):
    if args.data == 'java':
        data_path = const.JAVA_PATH
    else:
        data_path = const.SYNTH_PATH

    if args.labels_only:
        const.LABELS_ONLY = True

    if args.keep_duplicates:
        remove_duplicates = False
    else:
        remove_duplicates = True

    train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/', labels_only=const.LABELS_ONLY,
                                    build_language=False, remove_duplicates=remove_duplicates, to_tensors=False, sort=False,)
    test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/', labels_only=const.LABELS_ONLY,
                                   build_language=False, remove_duplicates=remove_duplicates, to_tensors=False, sort=False)
    valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/', labels_only=const.LABELS_ONLY,
                                    build_language=False, remove_duplicates=remove_duplicates, to_tensors=False, sort=False)

    if const.PREPROCESS_USE_BPE:
        print('Creating training files...')
        with open(const.PREPROCESS_TRAIN_PATH_DOC, 'w', encoding='utf-8') as train_file:
            for text in train_data.df['docstring_tokens']:
                train_file.write(f'{" ".join(text)}\n')
            for text in test_data.df['docstring_tokens']:
                train_file.write(f'{" ".join(text)}\n')
            for text in valid_data.df['docstring_tokens']:
                train_file.write(f'{" ".join(text)}\n')
        if not const.LABELS_ONLY:
            with open(const.PREPROCESS_TRAIN_PATH_CODE, 'w', encoding='utf-8') as train_file:
                for text in train_data.df['code_tokens']:
                    train_file.write(f'{" ".join(text)}\n')
                for text in test_data.df['code_tokens']:
                    train_file.write(f'{" ".join(text)}\n')
                for text in valid_data.df['code_tokens']:
                    train_file.write(f'{" ".join(text)}\n')

        print('Creating codes files...')
        with open(const.PREPROCESS_TRAIN_PATH_DOC, encoding='utf-8') as train_file, \
                open(const.PREPROCESS_CODES_PATH_DOC, 'w', encoding='utf-8') as codes_file:
            subword_nmt.learn_bpe(train_file, codes_file, const.PREPROCESS_VOCAB_SIZE_DOC)
        if not const.LABELS_ONLY:
            with open(const.PREPROCESS_TRAIN_PATH_CODE, encoding='utf-8') as train_file, \
                    open(const.PREPROCESS_CODES_PATH_CODE, 'w', encoding='utf-8') as codes_file:
                subword_nmt.learn_bpe(train_file, codes_file, const.PREPROCESS_VOCAB_SIZE_CODE)

        print('Creating vocab files...')
        with open(const.PREPROCESS_TRAIN_PATH_DOC, encoding='utf-8') as train_file, \
                open(const.PREPROCESS_VOCAB_PATH_DOC, 'w', encoding='utf-8') as vocab_file:
            subword_nmt.get_vocab(train_file, vocab_file)
        if not const.LABELS_ONLY:
            with open(const.PREPROCESS_TRAIN_PATH_CODE, encoding='utf-8') as train_file, \
                    open(const.PREPROCESS_VOCAB_PATH_CODE, 'w', encoding='utf-8') as vocab_file:
                subword_nmt.get_vocab(train_file, vocab_file)

        print('Applying codes...')
        with open(const.PREPROCESS_CODES_PATH_DOC, encoding='utf-8') as codes_file, \
                open(const.PREPROCESS_VOCAB_PATH_DOC, encoding='utf-8') as vocab_file:
            vocab = subword_nmt.read_vocabulary(vocab_file, const.PREPROCESS_VOCAB_FREQ_THRESHOLD)
            bpe = subword_nmt.BPE(codes_file, vocab=vocab)
            train_data.df[['docstring_tokens']] = train_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
            test_data.df[['docstring_tokens']] = test_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
            valid_data.df[['docstring_tokens']] = valid_data.df[['docstring_tokens']].applymap(bpe.segment_tokens)
        if not const.LABELS_ONLY:
            with open(const.PREPROCESS_CODES_PATH_CODE, encoding='utf-8') as codes_file, \
                    open(const.PREPROCESS_VOCAB_PATH_CODE, encoding='utf-8') as vocab_file:
                vocab = subword_nmt.read_vocabulary(vocab_file, const.PREPROCESS_VOCAB_FREQ_THRESHOLD)
                bpe = subword_nmt.BPE(codes_file, vocab=vocab)
                train_data.df[['code_tokens']] = train_data.df[['code_tokens']].applymap(bpe.segment_tokens)
                test_data.df[['code_tokens']] = test_data.df[['code_tokens']].applymap(bpe.segment_tokens)
                valid_data.df[['code_tokens']] = valid_data.df[['code_tokens']].applymap(bpe.segment_tokens)

    print('Working on dataframe...')
    if const.LABELS_ONLY:
        train_data.df[['code_tokens']] = train_data.df[['docstring_tokens']]
        test_data.df[['code_tokens']] = test_data.df[['docstring_tokens']]
        valid_data.df[['code_tokens']] = valid_data.df[['docstring_tokens']]

    print('Building languages...')
    input_lang = data.Lang('docstring')
    output_lang = data.Lang('code')
    train_data.build_language(languages=[input_lang, output_lang])
    test_data.build_language(languages=[input_lang, output_lang])
    valid_data.build_language(languages=[input_lang, output_lang])

    print('Converting to tensors...')
    train_data.to_tensors()
    test_data.to_tensors()
    valid_data.to_tensors()

    if not const.SHUFFLE_DATA:
        train_data.sort()
        test_data.sort()
        valid_data.sort()

    df_all = pd.concat([train_data.df, test_data.df, valid_data.df])

    print('Saving...')
    pickle.dump(train_data, open(const.TRAIN_DATA_SAVE_PATH, 'wb'))
    pickle.dump(test_data, open(const.TEST_DATA_SAVE_PATH, 'wb'))
    pickle.dump(valid_data, open(const.VALID_DATA_SAVE_PATH, 'wb'))
    pickle.dump(input_lang, open(const.INPUT_LANG_SAVE_PATH, 'wb'))
    pickle.dump(output_lang, open(const.OUTPUT_LANG_SAVE_PATH, 'wb'))
    df_all.to_pickle(const.ALL_DATA_DF_SAVE_PATH)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
