import openai
import pandas as pd
import puz
import datetime
import sys
import os
from dotenv import load_dotenv
from twilio.rest import Client
import uuid


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

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
            model="text-davinci-003",
            prompt=get_prompt(word),
            temperature=1,
        )
        clue = response.choices[0]['text']
        if not word.lower() in clue.lower():
            if 'Word:' in clue:
                clue = clue.split('Word:')[0]
            if 'Clue:' in clue:
                clue = clue.split('Clue:')[0]
            if clue.strip() != "":
                return clue


    assert False, "Could not generate clue for word: {}".format(word)

"""
Return a list of puzzle information where each element is the word,
the position the word starts, and if it is across or down
"""
def get_words_from_grid(grid):
    words = []
    rows = None
    if isinstance(grid, str):
        rows = grid.split("\n")
    else:
        rows = grid
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
    puzzle.author = "twitter: @noncompeteparty"
    puzzle.width = grid_width
    puzzle.height = grid_height
    puzzle.fill = '-' * grid_width * grid_height
    puzzle.solution = open("generated_data/cw_output.txt", "r").read().replace("\n", "").upper()
    puzzle.clues = cw_data["clue"].tolist()

    ## save the puz file
    puzzle.save("generated_data/crobot_"+str(datetime.date.today())+".puz")

def initial_generation(grid, grid_width, grid_height):
    ## get all words from the grid

    all_word_data = get_words_from_grid(grid)

    ## append a clue to each element in the list
    for word_info in all_word_data:
        word = word_info[0]
        print("Generating clue for word: {}".format(word))
        clue = generate_clue(word)
        word_info.append(clue)
    
    cw_data = pd.DataFrame(all_word_data, columns=["word", "start_pos", "direction", "clue"])
    return cw_data

def regenerate_clues(words_to_fix):
    cw_data = pd.read_csv("generated_data/cw_data.csv")
    for word in words_to_fix:
        print("Regenerating clue for word: {}".format(word))
        clue = generate_clue(word)
        cw_data.loc[cw_data["word"] == word, "clue"] = clue
    return cw_data

def main():
    grid = open("generated_data/cw_output.txt", "r").read()
    grid_width = len(grid.split("\n")[0])
    grid_height = len(grid.split("\n"))
    cw_data = None
    
    if len(sys.argv) == 1:
        cw_data = initial_generation(grid, grid_width, grid_height)
    else:
        words_to_fix = sys.argv[1:]
        cw_data = regenerate_clues(words_to_fix)
    print("CROSSWORD DATA")

    print(cw_data.to_string())
    # send the crossword data to my phone
    if len(cw_data.to_string < 1600):
        client = Client(os.environ['twilio_account_sid'], os.environ['twilio_auth_token']) 
        uid = str(uuid.uuid4())
        message = client.messages.create(  
                                    messaging_service_sid=os.environ['messaging_service_sid'], 
                                    body=cw_data.to_string() + '\n' + uid,      
                                    to=os.environ['sri_phone_number'])

    ## save crossword data as a csv
    cw_data.to_csv("generated_data/cw_data.csv", index=False)

    create_puz_file(cw_data, grid_width, grid_height)

if __name__ == "__main__":
    main()