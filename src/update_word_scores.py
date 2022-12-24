import pandas as pd
from upload_puz import get_grid
from generate_puz_file import get_words_from_grid
import puz
import datetime as dt
import os
from random import shuffle
import numpy as np

## split words dataframe into 100 parts and shuffle each part before returning each part
def split_shuffle_words(word_df):
    new_word_list = []
    current_word_list = word_df['word'].tolist()
    word_sections = np.array_split(current_word_list, 100)
    for section in word_sections:
        shuffle(section)
        new_word_list.extend(section)
    word_df['word'] = new_word_list
    return word_df

## return diction of word to score multiplier
def get_all_historical_words():
    my_path = "generated_data/"
    all_files = os.listdir(my_path)
    word_to_multiplier = {}
    for file in all_files:
        if file.startswith("crobot_"):
            p = puz.read(os.path.join(my_path, file))
            grid = get_grid(p)
            words = list(map(lambda x:x[0], get_words_from_grid(grid)))
            date_in_file = pd.to_datetime(file.split("_")[1].split(".")[0]).date()
            date_today = dt.date.today()
            days_since = (date_today - date_in_file).days

            MAX_DAYS = 20

            multiplier = max(min(MAX_DAYS, days_since-1),0)/MAX_DAYS

            for word in words:
                lword = word.lower()
                if not lword in word_to_multiplier:
                    word_to_multiplier[lword] = multiplier
                else:
                    word_to_multiplier[lword] = min(multiplier,word_to_multiplier[lword])
    return word_to_multiplier


def main():
    word_to_multiplier = get_all_historical_words()
    word_scores = pd.read_csv("raw_data/all_word_scores_no_title.csv", header=None, index_col=None)
    word_scores.columns = ["word", "score"]
    word_indexes = word_scores[word_scores['word'].isin(word_to_multiplier.keys())].index
    for index in word_indexes:
        word = word_scores.loc[index, "word"]
        score = int(word_scores.loc[index, 'score'] * word_to_multiplier[word])
        word_scores.at[index, 'score'] = score
    word_scores = split_shuffle_words(word_scores)
    word_scores.to_csv("raw_data/all_word_scores_new_scores.csv", header=None, index=None)

if __name__ == '__main__':
    main()