import openai
import pandas as pd
import puz
import datetime

OPENAI_API_KEY="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"
openai.api_key = OPENAI_API_KEY

N_ATTEMPTS = 5

def get_prompt(word: str):
    prompt = open('raw_data/gpt_prompt_monday.txt').read() +"""

Word: {}
Clue: """.format(
        word.upper()
    )
    return prompt


def generate_clue(word: str) -> str:
    for i in range(N_ATTEMPTS):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=get_prompt(word),
            temperature=.5,
        )
        clue = response.choices[0]['text']
        if not word.lower() in clue.lower():
            if 'Word:' in clue:
                return clue.split('Word:')[0]
            elif 'Clue:' in clue:
                return clue.split('Clue:')[0]
            else:
                return clue


    return ""

"""
Return a list of puzzle information where each element is the word,
the position the word starts, and if it is across or down
"""
def get_words_from_grid(grid):
    words = []
    rows = grid.split("\n")

    ## start with across
    for i in range(len(rows)):
        line = rows[i]
        word = ""
        start_pos = (i+1, 1)
        for j in range(len(line)):
            letter = line[j]
            if letter == ".":
                if word != "":
                    words.append([word, start_pos , "across"])
                    word = ""
                start_pos = (i+1,j+2) # +2 because its the one after the blk square
            else:
                word += letter
        if word != "":
            words.append([word, start_pos , "across"])

    ## now do down
    for j in range(len(rows[0])):
        word = ""
        start_pos = (1,j+1)
        for i in range(len(rows)):
            line = rows[i]
            if line[j] == ".":
                if word != "":
                    words.append([word, start_pos , "down"])
                    word = ""
                start_pos = (i+2,j+1) # +2 because its the one after the blk square
            else:
                word += line[j]
        if word != "":
            words.append([word, start_pos , "down"])

    return words

def create_puz_file(cw_data, grid_width, grid_height):
    cw_data = cw_data.sort_values(by=["start_pos", "direction"])
    cw_data = cw_data.reset_index(drop=True)

    ## create the puz file
    puzzle = puz.Puzzle()
    puzzle.title = "TODAY'S CROBOT " + str(datetime.date.today())
    puzzle.author = "twitter: @nocompeteparty"
    puzzle.width = grid_width
    puzzle.height = grid_height
    puzzle.fill = '-' * grid_width * grid_height
    puzzle.solution = open("generated_data/cw_output.txt", "r").read().replace("\n", "").upper()
    puzzle.clues = cw_data["clue"].tolist()

    ## save the puz file
    puzzle.save("generated_data/crobot_"+str(datetime.date.today())+".puz")

def main():
    ## get all words from the grid
    grid = open("generated_data/cw_output.txt", "r").read()
    grid_width = len(grid.split("\n")[0])
    grid_height = len(grid.split("\n"))

    all_word_data = get_words_from_grid(grid)

    ## append a clue to each element in the list
    for word_info in all_word_data:
        word = word_info[0]
        print("Generating clue for word: {}".format(word))
        clue = generate_clue(word)
        word_info.append(clue)
    
    cw_data = pd.DataFrame(all_word_data, columns=["word", "start_pos", "direction", "clue"])

    ## save crossword data as a csv
    cw_data.to_csv("generated_data/cw_data.csv", index=False)

    create_puz_file(cw_data, grid_width, grid_height)

if __name__ == "__main__":
    main()