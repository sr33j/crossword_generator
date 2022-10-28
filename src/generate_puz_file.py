import openai
import pandas as pd
import puz
import datetime

OPENAI_API_KEY="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"
openai.api_key = OPENAI_API_KEY

N_ATTEMPTS = 5

def get_prompt(word: str):
    prompt = """Suggest a concise, crossword clue for a word. 
    A crossword clue is a pithy riddle or definition where the word is the answer.
    A crossword clue cannot have any part of the word in it.
    A crossword clue has to make sense. The clue has to be a valid clue for the word.
    When someone hears the clue, they should think of the word.
    Several examples are listed below.

Word: EINE
Clue: A as in Aachen.

Word: AMNESIA
Clue: A name is troublesome.

Word: MUSH
Clue: Amundsen's Forwarding Address.

Word: INN
Clue: An overnight letter?

Word: RIVER
Clue: Bank depositor.

Word: STREAKS
Clue: Barely runs?

Word: ACE
Clue: Big heart.

Word: CHESS
Clue: Black and white set.

Word: TRAMPOLINE
Clue: Bouncer's place.

Word: HYPHENATE
Clue: Break one's word.

Word: TOASTERS
Clue: Browning pieces.

Word: BRAINWASH
Clue: Bust down reason?

Word: TARTAR
Clue: Calculus for canines?

Word: ASH
Clue: Camel's end?

Word: ONEAL
Clue: Celebrity center.

Word: BEAR IN MIND
Clue: Consider an imaginary animal?

Word: NOTRE
Clue: Dame's introduction.

Word: REINDEER
Clue: Dancer's group.

Word: ZIPCODE
Clue: Delivery aid.

Word: CATER
Clue: Do the dishes.

Word: COMBOVERS
Clue: Do's that are don'ts.

Word: ALE
Clue: Draft pick.

Word: TUV
Clue: Eight letters.

Word: ONE AFTER ANOTHER
Clue: Eleven?

Word: I AM
Clue: English sum.

Word: TAILS
Clue: Flip side?

Word: ADAM
Clue: Garden party.

Word: DEADENS
Clue: Gives a number to.

Word: IAGO
Clue: Globe plotter.

Word: MEALTIME
Clue: Grace period.

Word: KENNEDY
Clue: Half profile?

Word: RENAMED
Clue: Handled better?

Word: PREOP
Clue: Having not yet made the cut?

Word: LOO
Clue: Head of England.

Word: TINMAN
Clue: Heartless one?

Word: ANAGRAM
Clue: Horrid glances from Charles Grodin?

Word: LISP
Clue: How to make a sinner thinner.

Word: CARBON
Clue: It can help you get a date?

Word: BEDSHEET
Clue: It may be fit for a queen.

Word: TEN
Clue: It may be hung from a board.

Word: REFRAIN
Clue: It may come after a bridge.

Word: LIONEL
Clue: Its employees are in training?

Word: STEVE
Clue: Jobs in the computer biz.

Word: SHOW
Clue: Just a little out of place?

Word: HIP FLASK
Clue: Kick in the pants?

Word: PENTAGON
Clue: Large container of brass.

Word: TREE
Clue: Leaves home.

Word: BLEEP
Clue: Lift a curse.

Word: PET NAME
Clue: Love handle?

Word: MNO
Clue: LP insert.

Word: LIAR
Clue: Make-up artist.

Word: LATIN
Clue: Mass communication medium.

Word: UTOPIA
Clue: More work.

Word: JAPAN
Clue: No performers found here.

Word: NEON
Clue: Number 10 on a table.

Word: TORAH
Clue: Numbers holder.

Word: NERO
Clue: Octavia's offer?

Word: DUET
Clue: One can't do this.

Word: MOSES
Clue: One ordered to take two tablets.

Word: TRE
Clue: One past due?

Word: AUTOPILOTS
Clue: Ones who never think of flying?

Word: SEED
Clue: Open position

Word: MALE
Clue: Owner of the Y?

Word: SOLE
Clue: Oxford foundation?

Word: TEFLON
Clue: Pan films?

Word: ACRE
Clue: Part of a plot.

Word: LEAKED
Clue: Passed illegally.

Word: RELAY RACES
Clue: Passing events.

Word: WAVES
Clue: Permanent features

Word: DREIDEL
Clue: Place to find a nun?

Word: PIXEL
Clue: Point of resolution?

Word: ZLOTYS
Clue: Pole vault units?

Word: ASIA
Clue: Polo grounds.

Word: ROE
Clue: Preschoolers.

Word: REBELLED
Clue: Pretty Girl in Crimson Rose?

Word: CLOTHE
Clue: Put into gear.

Word: PAINTS
Clue: Puts on a coat.

Word: ANTHILL
Clue: Queen's home.

Word: SHEA
Clue: Queens plate setting.

Word: REGISTRAR
Clue: Record Holder.

Word: YOHOHO
Clue: Refrain from piracy.

Word: SATURN
Clue: Ring bearer.

Word: SOMERSAULT
Clue: Roast mules go topsy turvy.

Word: LORELEI
Clue: Rock Singer.

Word: LARGESS
Clue: S?

Word: BISHOPS
Clue: See people?

Word: DRAW
Clue: Select a tie?

Word: SHOPS
Clue: Selling points?

Word: EVICTS
Clue: Sends off letters.

Word: ANT
Clue: Six footer.

Word: EDITORIAL
Clue: Slanted column.

Word: NOTES
Clue: So and so?

Word: EPISODE
Clue: Soap unit.

Word: SHEETS
Clue: Some are fit for a king.

Word: AHAB
Clue: Starbuck's orderer.

Word: TONES
Clue: Steps on a scale?

Word: KNEE
Clue: Support for a proposal.

Word: SAWS
Clue: They go back and forth to work.

Word: TENSPEEDS
Clue: They may be shifted in transit.

Word: BOAS
Clue: They wrap their food well.

Word: ELLIPSIS
Clue: Three points in a row, perhaps.

Word: SNAKEEYES
Clue: Throw for a loss?

Word: ONEILL
Clue: Tip of Massachusetts.

Word: ESE
Clue: Tip of one's tongue.

Word: SHOE
Clue: Tongue-tied one.

Word: SYNONYM
Clue: United, for one.

Word: BALD SPOT
Clue: Unlocked area?

Word: ETCETERA
Clue: Used to avoid listing.

Word: LEAVE
Clue: Way off base.

Word: ONE
Clue: What I might mean.

Word: BUXOM
Clue: What the bronco does to chesty cowgirls?

Word: OFFICE
Clue: What they may be out of in January.

Word: ODE
Clue: Work with feet.

Word: CBS
Clue: Would one rather work there?

Word: {}
Clue: """.format(
        word.lower()
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
    puzzle.title = "TODAY'S CROBOT " + str(datetime.date.today())
    puzzle.author = "Non-Compete Party"
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