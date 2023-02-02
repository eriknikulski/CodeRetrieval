import itertools

import numpy as np
import torch
from dpu_utils.utils import RichPath
from dpu_utils.codeutils.deduplication import DuplicateDetector
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import const
import data
import pad_collate


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
        self.use_negatives = create_negatives
        self.lang = language
        self.items = ['docstring_tokens', 'code_sequence', 'methode_name', 'code_tokens']
        self.negative_items = []
        self.len = 0

        self.init_attributes(self.items)

        self.df = read_folder(RichPath.create(path))

        self.df[['docstring_tokens']] = self.df[['docstring_tokens']].applymap(transform)
        self.df[['code_sequence']] = self.df[['code']].applymap(target_transform)
        self.df[['methode_name']] = self.df[['func_name']].applymap(get_methode_name)
        self.df[['code_tokens']] = self.df[['code']].applymap(get_code_tokens)

        # Note: -1 because an EOS token is appended to the sequence later
        self.enforce_length_constraints(min_tokens_docstring, max_tokens_docstring - 1,
                                        min_tokens_code,  max_tokens_code - 1)
        if remove_duplicates:
            self.remove_duplicates()

        self.df.reset_index(drop=True, inplace=True)
        self.len = len(self.df)

        if self.negatives:
            self.create_negatives()

        if build_language:
            self.build_language()

        if to_tensors:
            self.to_tensors()

        print(f'{self.__len__()} elements loaded!\n')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        working_items = self.items
        if self.use_negatives:
            working_items +=  self.negative_items
        items = list(itertools.chain.from_iterable(zip(working_items, [elem + '_length' for elem in working_items])))
        return tuple(getattr(self, item)[idx] for item in items)

    def init_attributes(self, attr_list):
        for attr in attr_list:
            setattr(self, attr, None)

    def remove_duplicates(self):
        self.df = remove_duplicate_code_df(self.df)

    def create_negatives(self):
        self.negative_items =  ['neg_' + item for item in self.items]
        self.init_attributes(self.negative_items)

        if self.df is not None:
            for item, neg_item in zip(self.items, self.negative_items):
                self.df[[neg_item]] = self.df[[item]].sample(frac=1).reset_index(drop=True)
        else:
            for item, neg_item in zip(self.items, self.negative_items):
                tensor = getattr(self, item)
                rand_perm = torch.randperm(tensor.size(0))
                setattr(self, neg_item, tensor[rand_perm])
                setattr(self, neg_item + '_length', getattr(self, item + '_length')[rand_perm])

    def recreate_negatives(self):
        for k, v in vars(self).items():
            if k.startswith('neg_'):
                setattr(self, k, None)
        self.create_negatives()

    def build_language(self, language=None):
        print('building language dictionaries')
        self.lang = language if language else data.Lang('lang')

        self.df[['docstring_tokens']].applymap(self.lang.add_sequence)
        self.df[['code_sequence']].applymap(self.lang.add_sequence)

        self.lang.reduce_vocab(max_tokens=const.PREPROCESS_VOCAB_SIZE)

    def to_tensors(self):
        assert self.lang
        print('converting sequences to tensors')

        for item in self.items + self.negative_items:
            self.df[[item]] = self.df[[item]].applymap(self.lang.list_from_sequence)

        self.set_lengths()

        for attr in self.items + self.negative_items:
            max_length = getattr(self, attr + '_length').max()
            setattr(self, attr, torch.tensor(np.array([np.pad(row, (0, max_length - len(row)),
                                                              constant_values=const.PAD_TOKEN)
                                                       for row in self.df[attr]]), dtype=torch.long))
        self.df = None

    def set_lengths(self):
        items = self.items + self.negative_items
        lengths = [elem + '_length' for elem in items]

        for elem, elem_l in zip(items, lengths):
            setattr(self, elem_l, torch.tensor(self.df[[elem]].applymap(len).to_numpy().flatten()))


    def enforce_length_constraints(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                   max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                   min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        if self.df is not None:
            self._enforce_length_constraints_df(min_tokens_docstring, max_tokens_docstring,
                                                min_tokens_code, max_tokens_code)
        else:
            self._enforce_length_constraints_attr(min_tokens_docstring, max_tokens_docstring,
                                                  min_tokens_code, max_tokens_code)
            if self.negatives:
                self.recreate_negatives()

    def _enforce_length_constraints_df(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                       max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                       min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        self.df = self.df[self.items][
            (self.df['docstring_tokens'].map(len) <= max_tokens_docstring) &
            (self.df['docstring_tokens'].map(len) >= min_tokens_docstring)]
        self.df = self.df[self.items][
            (self.df['code_sequence'].map(len) <= max_tokens_code) &
            (self.df['code_sequence'].map(len) >= min_tokens_code)]
        self.len = len(self.df)

    def _enforce_length_constraints_attr(self, min_tokens_docstring=const.MIN_LENGTH_DOCSTRING,
                                         max_tokens_docstring=const.MAX_LENGTH_DOCSTRING,
                                         min_tokens_code=const.MIN_LENGTH_CODE, max_tokens_code=const.MAX_LENGTH_CODE):
        doc_size = self.docstring_tokens.size(1)
        summed = (self.docstring_tokens == const.PAD_TOKEN).sum(axis=1)
        doc_mask = (min_tokens_docstring <= doc_size - summed) & (doc_size - summed <= max_tokens_docstring)

        code_size = self.code_sequence.size(1)
        summed = (self.code_sequence == const.PAD_TOKEN).sum(axis=1)
        code_mask = (min_tokens_code <= code_size - summed) & (code_size - summed <= max_tokens_code)

        mask = doc_mask & code_mask
        for attr in self.items + [attr + '_length' for attr in self.items]:
            setattr(self, attr, getattr(self, attr)[mask])

        self.docstring_tokens = self.docstring_tokens[::, :max_tokens_docstring]
        self.code_sequence = self.code_sequence[::, :max_tokens_code]
        self.methode_name = self.methode_name[::, :max_tokens_code]
        self.code_tokens = self.code_tokens[::, :max_tokens_code]

        self.len = self.docstring_tokens.size(0)


if __name__ == "__main__":
    lang = data.Lang('lang')
    test_data = CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/',
                            remove_duplicates=False, language=lang, create_negatives=True)
    test_data.enforce_length_constraints(max_tokens_code=80)
    test_data.use_negatives = False
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=pad_collate.collate)

    doc_seq, doc_seq_length, code_seq, code_seq_length, methode_name, methode_name_length, code_tokens, code_tokens_length = \
        next(iter(test_dataloader))

    print(doc_seq)
    print(doc_seq_length)

    print(code_seq)
    print(code_seq_length)
    print(methode_name)
    print(methode_name_length)
    print(code_tokens)
    print(code_tokens_length)
