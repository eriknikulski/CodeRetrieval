from dpu_utils.utils import RichPath
from dpu_utils.codeutils.deduplication import DuplicateDetector
import pandas as pd
from tqdm import tqdm

import const
import data


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon code_tokens field in data."
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=10)
    filter_mask = df.apply(lambda x: dd.add_file(id=x.doc_id,
                                                 tokens=x.code_tokens),
                           axis=1)
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether or not code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df['doc_id'].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(f'Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows.')
    return df[filter_mask & exclusion_mask]


def read_folder(folder: RichPath):
    assert folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(
            list(f.read_as_jsonl(error_handling=lambda m, e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)


def load(file_path):
    # return read_folder(RichPath.create(file_path + 'test/')),\
    #     read_folder(RichPath.create(file_path + 'train/')),\
    #     read_folder(RichPath.create(file_path + 'valid/'))

    return read_folder(RichPath.create(file_path + 'test/')), None, None


def get_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df.filter(items=['docstring_tokens', 'code_tokens'])[(df.docstring_tokens.map(len) < const.MAX_LENGTH) &
                                                                (df.code_tokens.map(len) < const.MAX_LENGTH)]\
        .applymap(data.normalizeSeq)


def get():
    df_test, df_train, df_valid = load(const.PROJECT_PATH + const.JAVA_PATH)
    df_test['code_tokens'] = df_test['docstring_tokens']        # only use docstrings  TODO: remove when something works

    df_test = remove_duplicate_code_df(df_test)
    # df_train = remove_duplicate_code_df(df_train)
    # df_valid = remove_duplicate_code_df(df_valid)

    pairs = get_pairs(df_test)

    input = data.Lang('query')
    output = data.Lang('code')
    for pair in pairs.itertuples():
        for token in pair.docstring_tokens:
            input.addWord(token)
        for token in pair.code_tokens:
            output.addWord(token)

    return input, output, pairs.values.tolist()


if __name__ == "__main__":
    _, _, pairs = get()
    for i in range(10):
        print(pairs[i][0])
        print(pairs[i][1])
        print()
