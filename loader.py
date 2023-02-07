from dpu_utils.utils import RichPath
from dpu_utils.codeutils.deduplication import DuplicateDetector
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import const
import data


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve near duplicates based upon code_sequence field in data."""
    assert 'code_sequence' in df.columns.values, 'Data must contain field code_sequence'
    # assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=const.MIN_NUM_TOKENS)
    filter_mask = df.apply(lambda x: dd.add_file(id=x.doc_id,
                                                 tokens=x.code_sequence),
                           axis=1)
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df['doc_id'].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(f'Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows.')
    return df[filter_mask & exclusion_mask].drop('doc_id', axis=1)


def read_folder(folder: RichPath):
    assert folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(
            list(f.read_as_jsonl(error_handling=lambda m, e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)


class CodeDataset(Dataset):
    def __init__(self, path, transform=data.normalize_docstring, target_transform=data.transform_code_sequence,
                 get_methode_name=data.get_code_methode_name, get_code_tokens=data.get_code_tokens,
                 min_tokens_docstring=const.MIN_LENGTH_DOCSTRING, max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                 min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE, language=None,
                 build_language=True, to_tensors=True, remove_duplicates=True, create_negatives=True):
        self.path = path
        self.negatives = create_negatives
        self.lang = language if language else None
        self.working_items = ['docstring_tokens', 'docstring_tokens_length', 'code_sequence', 'code_sequence_length',
                              'methode_name', 'methode_name_length', 'code_tokens', 'code_tokens_length']

        self.df = read_folder(RichPath.create(path))
        self.df = self.df[:100]

        self.df[['docstring_tokens']] = self.df[['docstring_tokens']].applymap(transform)
        self.df[['code_sequence']] = self.df[['code']].applymap(target_transform)
        self.df[['methode_name']] = self.df[['func_name']].applymap(get_methode_name)
        self.df[['code_tokens']] = self.df[['code']].applymap(get_code_tokens)

        self.df[['docstring_tokens_length']] = self.df[['docstring_tokens']].applymap(len)
        self.df[['code_sequence_length']] = self.df[['code_sequence']].applymap(len)
        self.df[['methode_name_length']] = self.df[['methode_name']].applymap(len)
        self.df[['code_tokens_length']] = self.df[['code_tokens']].applymap(len)


        self.enforce_length_constraints(min_tokens_docstring, max_tokens_docstring, min_tokens_code,  max_tokens_code)
        if remove_duplicates:
            self.remove_duplicates()

        self.df.reset_index(drop=True, inplace=True)

        if self.negatives:
            self.create_negatives()

        if build_language:
            self.build_language()

        if to_tensors:
            self.to_tensors()

        print(f'{self.__len__()} elements loaded!\n')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: doc, doc_length, code_sequence, code_sequence_length, methode_name, methode_name_length,
                 code_tokens, url
        """
        return self.df[idx]

    def remove_duplicates(self):
        self.df = remove_duplicate_code_df(self.df)

    def create_negatives(self):
        self.working_items += ['neg_' + item for item in self.working_items]

        self.df[['neg_docstring_tokens']] = self.df[['docstring_tokens']].sample(frac=1).reset_index(drop=True)
        self.df[['neg_docstring_tokens_length']] = self.df[['neg_docstring_tokens']].applymap(len)

        self.df[['neg_code_sequence']] = self.df[['code_sequence']].sample(frac=1).reset_index(drop=True)
        self.df[['neg_code_sequence_length']] = self.df[['neg_code_sequence']].applymap(len)

        self.df[['neg_methode_name']] = self.df[['methode_name']].sample(frac=1).reset_index(drop=True)
        self.df[['neg_methode_name_length']] = self.df[['neg_methode_name']].applymap(len)

        self.df[['neg_code_tokens']] = self.df[['code_tokens']].sample(frac=1).reset_index(drop=True)
        self.df[['neg_code_tokens_length']] = self.df[['code_tokens']].applymap(len)

    def build_language(self, language=None):
        print('building language dictionaries')
        self.lang = language if language else data.Lang('lang')

        self.df[['docstring_tokens']].applymap(self.lang.add_sequence)
        self.df[['code_sequence']].applymap(self.lang.add_sequence)

        self.lang.reduce_vocab(max_tokens=const.PREPROCESS_VOCAB_SIZE)

    def to_tensors(self):
        assert self.lang
        print('converting sequences to tensors')
        self.df[['docstring_tokens']] = self.df[['docstring_tokens']].applymap(self.lang.tensor_from_sequence)
        self.df[['code_sequence']] = self.df[['code_sequence']].applymap(self.lang.tensor_from_sequence)
        self.df[['code_tokens']] = self.df[['code_tokens']].applymap(self.lang.tensor_from_sequence)
        self.df[['methode_name']] = self.df[['methode_name']].applymap(self.lang.tensor_from_sequence)
        if self.negatives:
            self.df[['neg_docstring_tokens']] = self.df[['neg_docstring_tokens']].applymap(self.lang.tensor_from_sequence)
            self.df[['neg_code_sequence']] = self.df[['neg_code_sequence']].applymap(self.lang.tensor_from_sequence)
            self.df[['neg_code_tokens']] = self.df[['neg_code_tokens']].applymap(self.lang.tensor_from_sequence)
            self.df[['neg_methode_name']] = self.df[['neg_methode_name']].applymap(self.lang.tensor_from_sequence)

        self.df[['docstring_tokens_length']] = self.df[['docstring_tokens']].applymap(len)
        self.df[['code_sequence_length']] = self.df[['code_sequence']].applymap(len)
        self.df[['code_tokens_length']] = self.df[['code_tokens']].applymap(len)
        self.df[['methode_name_length']] = self.df[['methode_name']].applymap(len)
        if self.negatives:
            self.df[['neg_docstring_tokens_length']] = self.df[['neg_docstring_tokens']].applymap(len)
            self.df[['neg_code_sequence_length']] = self.df[['neg_code_sequence']].applymap(len)
            self.df[['neg_code_tokens_length']] = self.df[['neg_code_tokens']].applymap(len)
            self.df[['neg_methode_name_length']] = self.df[['neg_methode_name']].applymap(len)

        self.to_list()

    def to_list(self):
        print('convert dataframe to numpy')
        self.df = self.df.to_numpy().tolist()

    def enforce_length_constraints(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                   max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                   min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        if isinstance(self.df, pd.DataFrame):
            self._enforce_length_constraints_df(min_tokens_docstring, max_tokens_docstring,
                                                min_tokens_code, max_tokens_code)
        else:
            self._enforce_length_constraints_list(min_tokens_docstring, max_tokens_docstring,
                                                  min_tokens_code, max_tokens_code)

    def _enforce_length_constraints_df(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                       max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                       min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        self.df = self.df[self.working_items][
            (self.df['docstring_tokens'].map(len) <= max_tokens_docstring) &
            (self.df['docstring_tokens'].map(len) >= min_tokens_docstring)]
        self.df = self.df[self.working_items][
            (self.df['code_sequence'].map(len) <= max_tokens_code) &
            (self.df['code_sequence'].map(len) >= min_tokens_code)]

    def _enforce_length_constraints_list(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                         max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                         min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        doc_idx = self.working_items.index('docstring_tokens')
        code_idx = self.working_items.index('code_sequence')

        self.df = [elems for elems in self.df if
                   min_tokens_docstring <= len(elems[doc_idx]) <= max_tokens_docstring and
                   min_tokens_code <= len(elems[code_idx]) <= max_tokens_code]


if __name__ == "__main__":
    lang = data.Lang('lang')
    test_data = CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/', language=lang, create_negatives=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    doc_seq, doc_seq_length, code_seq, code_seq_length, methode_name, methode_name_length, code_tokens, code_tokens_length = \
        next(iter(test_dataloader))

    print(doc_seq)
    print(doc_seq_length)
    print(code_seq)
    print(code_seq_length)
