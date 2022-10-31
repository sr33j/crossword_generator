import pandas as pd

def main():
    last_cw_data = pd.read_csv("generated_data/cw_data.csv")
    last_words = last_cw_data["word"].tolist()
    word_scores = pd.read_csv("raw_data/all_word_scores_no_title.csv", header=None, index_col=None)
    word_scores.columns = ["word", "score"]
    word_indexes = word_scores[word_scores['word'].isin(last_words)].index
    for index in word_indexes:
        word_scores.at[index, 'score'] = 0
    word_scores.to_csv("raw_data/all_word_scores_new_scores.csv", header=None, index=None)

if __name__ == '__main__':
    main()