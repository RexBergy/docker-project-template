from pathlib import Path

from flask import json
import pandas as pd
import requests


if __name__ == "__main__":
    # Load dataset
    play_by_play_path = Path(__file__).parent.parent / "ift6758" / "ift6758" / "data" / "dataframe_2016_to_2019.csv"
    play_by_play = pd.read_csv(play_by_play_path).dropna()
    X = play_by_play[["shotDistance"]]
    #print(X.iloc[0])
    # r = requests.post(
	# "http://0.0.0.0:8000/predict", 
	# json=X.to_json()
    # )
    r = requests.post(
	"http://0.0.0.0:8000/download_registry_model", 
	json=json.dumps({"workspace": "philippe-bergeron-7-universit-de-montr-al-org/wandb-registry-model", 
                  "model": "Logistic regression", 
                  "version": 4})
    )
    print(r.json())

    #print(pd.read_json(X.to_json()))