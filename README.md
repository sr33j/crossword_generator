# crobot
auto generated crossowords

# Steps to run
- conda create --name crobot python=3.8
- conda activate crobot
- pip install -r requirements.txt
- define the variable in .env.example in your own .env file
- run ./full_flow.sh (might need to run `chmod u+x full_flow.sh` beforehand)
- if you dont like some clues you can run `python generate_puz_file.py word1 word2 ... wordn` for all the words you dont like the clues for
- once you like the puzzle, you can run `python upload_puz.py`
