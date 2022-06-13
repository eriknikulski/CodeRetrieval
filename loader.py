from dpu_utils.utils import RichPath
from dpu_utils.codeutils.deduplication import DuplicateDetector
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import const
import data


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon code_tokens field in data."
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    # assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=const.MIN_NUM_TOKENS)
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


class CodeDataset(Dataset):
    def __init__(self, path, transform=data.normalizeDocstring, target_transform=data.normalizeSeq,
                 max_tokens=const.MAX_LENGTH, min_tokens=const.MIN_LENGTH, labels_only=False, build_language=True,
                 to_tensors=True, remove_duplicates=True):
        self.path = path
        self.input_lang = None
        self.output_lang = None
        self.labels_only = labels_only

        self.df = read_folder(RichPath.create(path))
        self.df[['docstring_tokens']] = self.df[['docstring_tokens']].applymap(transform)
        self.df[['code_tokens']] = self.df[['code_tokens']].applymap(target_transform)
        self.df = self.df.filter(items=['docstring_tokens', 'code_tokens', 'url'])[
            (self.df.docstring_tokens.map(len) <= max_tokens) & (self.df.docstring_tokens.map(len) >= min_tokens)]
        if labels_only:
            self.df['code_tokens'] = self.df['docstring_tokens']
        if remove_duplicates:
            self.remove_duplicates()

        if build_language:
            self.build_language()

        if to_tensors:
            self.to_tensors()

        print(f'{self.__len__()} elements loaded!\n')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.to_numpy()[idx][:3].tolist()

    def get_langs(self):
        return self.input_lang, self.output_lang

    def remove_duplicates(self):
        self.df = remove_duplicate_code_df(self.df)

    def build_language(self):
        print('building language dictionaries')
        self.input_lang = data.Lang('docstring')
        self.output_lang = data.Lang('code')
        for pair in self.df.itertuples():
            self.input_lang.addSequence(pair.docstring_tokens)
            if not self.labels_only:
                self.output_lang.addSequence(pair.code_tokens)

    def to_tensors(self):
        assert self.input_lang and self.output_lang
        print('converting sequences to tensors')
        self.df[['docstring_tokens']] = self.df[['docstring_tokens']].applymap(
            lambda x: self.input_lang.tensorFromSequence(x))
        self.df[['code_tokens']] = self.df[['code_tokens']].applymap(
            lambda x: self.output_lang.tensorFromSequence(x))


if __name__ == "__main__":
    test_data = CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    test_labels, test_features, url = next(iter(test_dataloader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_labels.size()}")

    print(test_features)
    print(test_labels)
    print(url)
