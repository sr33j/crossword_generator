import requests
import json
import puz
import tweepy
import pandas as pd
import datetime
import os
from dotenv import load_dotenv


def tweet(new_pid):
    load_dotenv()

    TWITTER_API_KEY = os.environ['TWITTER_API_KEY']
    TWITTER_API_SECRET = os.environ['TWITTER_API_SECRET']
    TWITTER_API_ACCESS_TOKEN = os.environ['TWITTER_API_ACCESS_TOKEN']
    TWITTER_API_ACCESS_SECRET = os.environ['TWITTER_API_ACCESS_SECRET']

    
    auth = tweepy.OAuthHandler(TWITTER_API_KEY,TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_API_ACCESS_TOKEN, TWITTER_API_ACCESS_SECRET)
    
    api = tweepy.API(auth)
    
    try:
        api.verify_credentials()
        print('Successful connection')
    except:
        print('Failed connection')

    new_tweet = "A new auto-generated mini!!! https://downforacross.com/beta/play/" + str(new_pid)
    api.update_status(new_tweet)

def get_last_pid():
    url = "https://api.foracross.com/api/puzzle_list?page=0&pageSize=50&filter%5BnameOrTitleFilter%5D=&filter%5BsizeFilter%5D%5BMini%5D=true&filter%5BsizeFilter%5D%5BStandard%5D=true"
    r = requests.get(url)
    data = json.loads(r.content)
    last_pid = int(data["puzzles"][0]["pid"])
    return last_pid

def get_grid(puzzle):
    index = 0
    grid = []
    for h in range(puzzle.height):
        row = []
        for w in range(puzzle.width):
            row.append(puzzle.solution[index])
            index += 1
        grid.append(row)
    return grid

def get_clues(puzzle_data):
    puzzle_data = pd.read_csv("generated_data/cw_data.csv")
    puzzle_data = puzzle_data.sort_values(by='start_pos')
    puzzle_data = puzzle_data.reset_index(drop=True)

    across_list = [None for i in range(0, len(puzzle_data) + 1)]
    down_list = [None for i in range(0, len(puzzle_data) + 1)]

    square_no = 0
    last_square = None
    for i, row in puzzle_data.iterrows():
        if row['start_pos'] != last_square:
            square_no +=1
            last_square = row['start_pos']
        if row['direction'] == 'across':
            across_list[square_no] = row['clue']
        else:   
            down_list[square_no] = row['clue']
    return across_list, down_list

def upload_to_downforacross(puzzle, puzzle_data):
    grid = get_grid(puzzle)
    across_list, down_list = get_clues(puzzle_data)
    new_pid = get_last_pid() + 10

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        # Already added when you pass json=
        # 'Content-Type': 'application/json',
        'Origin': 'https://downforacross.com',
        'Referer': 'https://downforacross.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    json_data = {
        'puzzle': {
            'grid': grid,
            'circles': [],
            'shades': [],
            'info': {
                'type': 'Mini Puzzle',
                'title': puzzle.title,
                'author': puzzle.author,
                'description': '',
            },
            'clues': {
                'across': across_list,
                'down': down_list,
            },
            'private': False,
        },
        'pid': new_pid,
        'isPublic': True,
    }

    response = requests.post('https://api.foracross.com/api/puzzle', headers=headers, json=json_data)
    print(response.text)
    return new_pid

def main():
    p = puz.read("generated_data/crobot_"+str(datetime.date.today())+".puz")
    puzzle_data = pd.read_csv("generated_data/cw_data.csv")
    new_pid = upload_to_downforacross(p, puzzle_data)
    tweet(new_pid)

if __name__ == "__main__":
    main()