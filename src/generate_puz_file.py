import openai
import pandas as pd
import puz

OPENAI_API_KEY="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"
openai.api_key = OPENAI_API_KEY

N_ATTEMPTS = 5

def get_prompt(word: str):
    prompt = """Suggest a concise, crossword clue for a word

Word: area
Clue: Geometry calculation
Word: ipad
Clue: Fire tablet competitor
Word: india
Clue: Where rajahs once ruled
Word: {}
Clue:""".format(
        word.lower()
    )
    return prompt


def generate_clue(word: str) -> str:
    for i in range(N_ATTEMPTS):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=get_prompt(word),
            temperature=1,
        )
        clue = response.choices[0]['text']
        if not word in clue.lower():
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
    if word != "":
        words.append([word, start_pos , "down"])

    return words

def create_puz_file(cw_data, grid_width, grid_height):
    cw_data = cw_data.sort_values(by=["start_pos", "direction"])
    cw_data = cw_data.reset_index(drop=True)

    ## create the puz file
    puzzle = puz.Puzzle()
    puzzle.title = "Gen #1"
    puzzle.author = "Non-Compete Party"
    puzzle.width = grid_width
    puzzle.height = grid_height
    puzzle.fill = '-' * grid_width * grid_width
    puzzle.solution = open("cw_output.txt", "r").read().replace("\n", "").upper()
    puzzle.clues = cw_data["clue"].tolist()

    ## save the puz file
    puzzle.save("gen1.puz")

def main():
    ## get all words from the grid
    grid = open("cw_output.txt", "r").read()
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
    cw_data.to_csv("cw_data.csv", index=False)

    create_puz_file(cw_data, grid_width, grid_height)


if __name__ == "__main__":
    main()