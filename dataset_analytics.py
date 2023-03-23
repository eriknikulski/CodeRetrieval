import const
import data
from loader import CodeDataset


def analyse(path):
    lang = data.Lang('lang')
    dataset = CodeDataset(path, language=lang, remove_duplicates=False, verbose=True)


def main():
    print('\n\nAnalysing TEST data set....')
    analyse(const.PROJECT_PATH + const.JAVA_PATH + 'test/')

    print('\n\nAnalysing VALID data set....')
    analyse(const.PROJECT_PATH + const.JAVA_PATH + 'valid/')

    print('\n\nAnalysing TRAIN data set....')
    analyse(const.PROJECT_PATH + const.JAVA_PATH + 'train/')


if __name__ == '__main__':
    main()