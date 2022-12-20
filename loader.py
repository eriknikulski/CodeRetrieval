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


class CodeDataset(Dataset):
    def __init__(self, path, transform=data.normalize_docstring, target_transform=data.transform_code_sequence,
                 get_methode_name=data.get_code_methode_name, get_code_tokens=data.get_code_tokens,
                 min_tokens_docstring=const.MIN_LENGTH_DOCSTRING, max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                 min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE, labels_only=False,
                 languages=None, build_language=True, to_tensors=True, remove_duplicates=True, sort=False):
        self.path = path
        self.input_lang = None
        self.output_lang = None
        if languages:
            self.set_languages(languages)
        self.labels_only = labels_only
        self.working_items = ['docstring_tokens', 'docstring_tokens_length', 'code_sequence', 'code_sequence_length',
                              'methode_name', 'methode_name_length', 'code_tokens']

        self.df = read_folder(RichPath.create(path))
        self.df.loc[:, 'docstring_tokens'] = self.df.loc[:, 'docstring_tokens'].copy().map(transform)
        if labels_only:
            self.df.loc[:, 'code_sequence'] = self.df.loc[:, 'docstring_tokens'].copy()
            min_tokens_code = min_tokens_docstring
            max_tokens_code = max_tokens_docstring
        else:
            self.df.loc[:, 'code_sequence'] = self.df.loc[:, 'code'].copy().map(target_transform)

        self.df.loc[:, 'docstring_tokens_length'] = self.df.loc[:, 'docstring_tokens'].copy().map(len)
        self.df.loc[:, 'code_sequence_length'] = self.df.loc[:, 'code_sequence'].copy().map(len)

        self.df.loc[:, 'methode_name'] = self.df.loc[:, 'code'].copy().map(get_methode_name)
        self.df.loc[:, 'methode_name_length'] = self.df.loc[:, 'methode_name'].copy().map(len)

        self.df.loc[:, 'code_tokens'] = self.df.loc[:, 'code'].copy().map(get_code_tokens)

        self.enforce_length_constraints(min_tokens_docstring, max_tokens_docstring, min_tokens_code,  max_tokens_code)
        if remove_duplicates:
            self.remove_duplicates()

        if build_language:
            self.build_language()

        if to_tensors:
            self.to_tensors()

        if sort:
            self.sort()

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

    def get_langs(self):
        return self.input_lang, self.output_lang

    def remove_duplicates(self):
        self.df = remove_duplicate_code_df(self.df)
        self.df.drop('doc_id', axis=1, inplace=True)

    def build_language(self, languages=None):
        print('building language dictionaries')
        if languages:
            self.set_languages(languages)

        self.df.loc[:, 'docstring_tokens'].map(self.input_lang.add_sequence)
        self.df.loc[:, 'code_sequence'].map(self.output_lang.add_sequence)

        self.input_lang.reduce_vocab(max_tokens=const.PREPROCESS_VOCAB_SIZE_CODE)
        self.output_lang.reduce_vocab(max_tokens=const.PREPROCESS_VOCAB_SIZE_CODE)

    def set_languages(self, languages):
        self.input_lang = languages[0]
        self.output_lang = languages[1]

    def to_tensors(self):
        assert self.input_lang and self.output_lang
        print('converting sequences to tensors')
        self.df.loc[:, 'docstring_tokens'] = self.df.loc[:, 'docstring_tokens'].map(
            self.input_lang.tensor_from_sequence)
        self.df.loc[:, 'code_sequence'] = self.df.loc[:, 'code_sequence'].map(self.output_lang.tensor_from_sequence)
        self.df.loc[:, 'code_tokens'] = self.df.loc[:, 'code_tokens'].map(self.output_lang.tensor_from_sequence)
        self.df.loc[:, 'methode_name'] = self.df.loc[:, 'methode_name'].map(self.output_lang.tensor_from_sequence)

        self.df.loc[:, 'docstring_tokens_length'] = self.df.loc[:, 'docstring_tokens'].copy().map(len)
        self.df.loc[:, 'code_sequence_length'] = self.df.loc[:, 'code_sequence'].copy().map(len)
        self.df.loc[:, 'methode_name_length'] = self.df.loc[:, 'methode_name'].copy().map(len)

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
        self.df = self.df.loc[:, self.working_items][
            (self.df.loc[:, 'docstring_tokens'].map(len) <= max_tokens_docstring) &
            (self.df.loc[:, 'docstring_tokens'].map(len) >= min_tokens_docstring)]
        self.df = self.df.loc[:, self.working_items][
            (self.df.loc[:, 'code_sequence'].map(len) <= max_tokens_code) &
            (self.df.loc[:, 'code_sequence'].map(len) >= min_tokens_code)]

    def _enforce_length_constraints_list(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                         max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                         min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        doc_idx = self.working_items.index('docstring_tokens')
        code_idx = self.working_items.index('code_sequence')

        return [elems for elems in self.df if
                min_tokens_docstring <= len(elems[doc_idx]) <= max_tokens_docstring and
                min_tokens_code <= len(elems[code_idx]) <= max_tokens_code]

    def sort(self):
        self.df.sort_values(by=['code_sequence', 'docstring_tokens'], key=lambda x: x.apply(len), inplace=True)


if __name__ == "__main__":
    input_lang = data.Lang('docstring')
    output_lang = data.Lang('code')
    test_data = CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/', languages=[input_lang, output_lang])
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    doc_seq, doc_seq_length, code_seq, code_seq_length, methode_name, methode_name_length, code_tokens = \
        next(iter(test_dataloader))

    print(doc_seq)
    print(doc_seq_length)
    print(code_seq)
    print(code_seq_length)
