from pathlib import Path

from flask import json
import pandas as pd
import requests

from  game_client import GameClient




if __name__ == "__main__":


    gc = GameClient()
    # first ping
    all_events = gc.get_game_and_filter_from_json("/Users/philippebergeron/Documents/Universite/automne2024/ift6758/docker-project-template/ift6758/ift6758/client/game1.json")
    print(all_events)

    # second ping
    two_events = gc.get_game_and_filter_from_json("/Users/philippebergeron/Documents/Universite/automne2024/ift6758/docker-project-template/ift6758/ift6758/client/game2.json")
    print(two_events)
