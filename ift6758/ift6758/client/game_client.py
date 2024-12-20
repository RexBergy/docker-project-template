from collections import defaultdict
import json
import requests
import pandas as pd
import logging
import numpy as np


logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        self.pointers = defaultdict(int)
        self.pointer = 0

    def get_game_and_filter(self, game_id: int) -> pd.DataFrame:
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        response = requests.get(url)
        data = None
        if response.status_code == 200:

            data = response.json()

        else:
            print("Failed")

        if data is None:
            return
        
        all_plays = data['plays']
        try:
            filtered_plays = all_plays[self.pointers[game_id]:]
            self.pointers[game_id] = len(all_plays)

            data['plays'] = filtered_plays

            return df_convert(data)
        except:
            return 
        
    def get_game_and_filter_from_json(self, json_game: str) -> pd.DataFrame:
        with open(json_game) as f:
            data = json.load(f)
        
        all_plays = data['plays']
        try:
            #print(self.pointer)
            filtered_plays = all_plays[self.pointer:]
            self.pointer = len(all_plays)
            
            data['plays'] = filtered_plays

            return df_convert(data)
        except:
            return 

def get_coor(row: pd.Series, home_team_initial_side: str) -> list:
    initial_side = None

    if(row['teamSide'] == 'home'):
        initial_side = home_team_initial_side
    else:
        if home_team_initial_side == 'left':
            initial_side = 'right'
        else:
            initial_side = 'left'


    new_coords = [(0, 0), (0,0)]
    current_side = initial_side
    if str(row['idGame'])[4:6] == '02' and row['numberPeriod'] <= 3:
        if row['numberPeriod'] % 2 == 0:
            # on change le camp
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'
    else:
        if row['numberPeriod'] % 2 == 0:
            if initial_side == 'left':
                current_side = 'right'
            else:
                current_side = 'left'

    if current_side == 'left':
        new_coords = [(-row['yCoord'], row['xCoord']), (-row['previousYCoord'], row['previousXCoord'])]
    else:
        new_coords = [(row['yCoord'], -row['xCoord']), (row['previousYCoord'], -row['previousXCoord'])]

    return new_coords

def v_angle(v1: np.array, v2: np.array) -> float:
    
    dot_product = np.dot(v1, v2)

    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_angle = dot_product / (norm_v1 * norm_v2)

    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def zoneshoot(clean_df: pd.DataFrame) -> pd.DataFrame:

    
    coords_list = [] 
    coords_last_event_list = []

    
    first_home_team_offensive_event = clean_df[(clean_df['zoneShoot'] == 'O') & (clean_df['teamSide'] == 'home')].iloc[
        0]
    home_team_initial_side = 'right' if first_home_team_offensive_event['xCoord'] < 0 else 'left'

    for _, row in clean_df.iterrows():
        new_coords, new_last_event_coord = get_coor(row, home_team_initial_side)
        coords_list.append(new_coords) 
        coords_last_event_list.append(new_last_event_coord)

    clean_df['adjustedCoord'] = coords_list
    clean_df['adjustedLastEventCoord'] = coords_last_event_list

    dist_euclidian = lambda x1, x2: np.round(np.linalg.norm(np.array(x1) - np.array(x2)), decimals=1)

    clean_df['shotDistance'] = clean_df.apply(lambda x: dist_euclidian(x['adjustedCoord'], np.array([0, 89])), axis=1)

    clean_df['distanceFromLastEvent'] = clean_df.apply(
        lambda x: dist_euclidian(x1=(x['xCoord'], x['yCoord'])
        , x2=(x['previousXCoord'], x['previousYCoord']))
        if not pd.isnull(x['previousXCoord']) else None, axis=1)
    clean_df['rebound'] = clean_df.apply(lambda x:
    True if x['previousEventType'] == 'shot-on-goal' else False, axis=1
    )

    # Add speed
    clean_df['speedFromLastEvent'] = clean_df.apply(lambda x:
    x['distanceFromLastEvent'] / x['timeSinceLastEvent']
    if x['timeSinceLastEvent'] != 0 else 0
    , axis=1)

    clean_df['shotAngle'] = clean_df.apply(
        lambda x: v_angle(x['adjustedCoord'] - np.array([0, 89]), np.array([0, -89])), axis=1)

    clean_df['reboundAngleShot'] = clean_df.apply(
        lambda x: v_angle(x['adjustedLastEventCoord'] - np.array([0, 89]), np.array([0, -89]) + x['shotAngle']
        if x['rebound'] else 0), axis=1)

    clean_df.drop(columns=['adjustedCoord'], inplace=True)
    clean_df.drop(columns=['adjustedLastEventCoord'], inplace=True)

    clean_df['offensivePressureTime'] = clean_df.groupby('eventOwnerTeam')['gameSeconds'].diff()

    # Convert the time to minutes and seconds
    clean_df['offensivePressureTime'] = clean_df.apply(lambda x: 0
    if pd.isnull(x['offensivePressureTime']) else x['offensivePressureTime'], axis=1)

    return clean_df

def empty_goal_func(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda x: x['situationCode'][3] if x['teamSide'] == 'away' else x['situationCode'][0] if len(x['situationCode']) == 4 else 0
                    , axis=1).map(
        {'0': True, '1': False})


def goal_situation(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda x: "Advantage" if (int(x['situationCode'][1]) > int(x['situationCode'][2]) and x[
        'teamSide'] == 'away') or (int(x['situationCode'][2]) > int(x['situationCode'][1]) and x[
        'teamSide'] == 'home')
    else "Disadvantage" if (int(x['situationCode'][1]) < int(x['situationCode'][2]) and x['teamSide'] == 'away') or
                           (int(x['situationCode'][2]) < int(x['situationCode'][1]) and x['teamSide'] == 'home')
    else 'Neutral', axis=1)


def time_convert(df: pd.DataFrame, column: str) -> pd.Series:

    df['minutes'] = df[column].str.split(':').str[0].astype(int)
    df['seconds'] = df[column].str.split(':').str[1].astype(int)
    df['numberPeriod'] = df['numberPeriod'].astype(int)

    df[column] = df['minutes'] * 60 + df['seconds'] + 20 * 60 * (df['numberPeriod'] - 1)

    df.drop(['minutes', 'seconds'], axis=1, inplace=True)

    return df[column]

def get_player(game_nhl: dict) -> pd.DataFrame:
    
    df_players = pd.DataFrame(game_nhl['rosterSpots'])[['playerId', 'firstName', 'lastName']]
    df_players['firstName'] = df_players['firstName'].apply(lambda x: x['default'])
    df_players['lastName'] = df_players['lastName'].apply(lambda x: x['default'])
    return df_players


def get_teams(game_nhl: dict) -> pd.DataFrame:
    home_team = {'teamId': game_nhl['homeTeam']['id'], 'teamName': game_nhl['homeTeam']['commonName']['default'],
                 'teamSide': 'home'}
    away_team = {'teamId': game_nhl['awayTeam']['id'], 'teamName': game_nhl['awayTeam']['commonName']['default'],
                 'teamSide': 'away'}
    return pd.DataFrame([home_team, away_team])

def ing_period(df: pd.DataFrame) -> pd.DataFrame:

    df_period = pd.DataFrame(df['periodDescriptor'].tolist())

    df_period[['number', 'maxRegulationPeriods']] = df_period[['number', 'maxRegulationPeriods']].astype(str)

    df_period['numberPeriod'] = df_period['number']

    return df_period


def ing_event(df: pd.DataFrame, df_players: pd.DataFrame) -> pd.DataFrame:
    df_details = pd.DataFrame(df['details'].tolist())

    df_details['shootingPlayerId'] = df_details['shootingPlayerId'].fillna(0) + df_details['scoringPlayerId'].fillna(0)

    df_details['goalieInNetId'] = df_details['goalieInNetId'].fillna(0)

    df_details['shootingPlayerId'] = df_details['shootingPlayerId'].astype(int)
    df_details['goalieInNetId'] = df_details['goalieInNetId'].astype('Int64')  

    df_details = pd.merge(df_players, df_details, left_on='playerId', right_on='shootingPlayerId', how='right').drop(
        columns=['playerId'])
    df_details['shootingPlayer'] = df_details['firstName'] + ' ' + df_details['lastName']
    df_details.drop(['firstName', 'lastName'], axis=1, inplace=True)

    df_details = pd.merge(df_players, df_details, left_on='playerId', right_on='goalieInNetId', how='right').drop(
        columns=['playerId'])

    df_details['goaliePlayer'] = df_details['firstName'] + ' ' + df_details['lastName']
    df_details.drop(['firstName', 'lastName'], axis=1, inplace=True)

    return df_details

def ing_event_bef(df: pd.DataFrame):
    df_copy = df.copy().shift(1)
    df['previousEventType'] = df_copy['typeDescKey']

    df['timeSinceLastEvent'] = df['gameSeconds'].diff()
    df['timeSinceLastEvent'] = df.apply(lambda x: 0
    if pd.isnull(x['timeSinceLastEvent']) else abs(x['timeSinceLastEvent']), axis=1)

    details = df_copy['details'].apply(pd.Series)
    df["previousXCoord"] = details['xCoord']
    df["previousYCoord"] = details['yCoord']

    return df

def df_convert(game_nhl: dict) -> pd.DataFrame:

    df_pbp = pd.DataFrame(game_nhl['plays'])

    df_players = get_player(game_nhl)
    df_teams = get_teams(game_nhl)

    clean_df = pd.DataFrame(df_pbp[['periodDescriptor', 'timeInPeriod', 'situationCode',
                                    'typeDescKey', 'details']])

    df_period = ing_period(clean_df)
    clean_df.drop('periodDescriptor', axis=1, inplace=True)

    clean_df.insert(0, 'idGame', game_nhl['id'])
    clean_df.insert(1, 'periodType', df_period['periodType'])
    clean_df.insert(3, 'numberPeriod', df_period['numberPeriod'])

    clean_df['gameSeconds'] = time_convert(clean_df, 'timeInPeriod')
    clean_df.drop('timeInPeriod', axis=1, inplace=True)

    clean_df = ing_event_bef(clean_df)

    clean_df = clean_df[(clean_df['typeDescKey'] == 'shot-on-goal') | (clean_df['typeDescKey'] == 'goal')].reset_index(
        drop=True)

    df_details = ing_event(clean_df, df_players)
    clean_df.drop('details', axis=1, inplace=True)

    df_details = pd.merge(df_teams, df_details, left_on='teamId', right_on='eventOwnerTeamId', how='right')

    clean_df['xCoord'] = df_details['xCoord']
    clean_df['yCoord'] = df_details['yCoord']
    clean_df['zoneShoot'] = df_details['zoneCode']
    clean_df['shootingPlayer'] = df_details['shootingPlayer']
    clean_df['goaliePlayer'] = df_details['goaliePlayer']
    clean_df['shotType'] = df_details['shotType']
    clean_df.insert(5, 'eventOwnerTeam', df_details['teamName'])
    clean_df['teamSide'] = df_details['teamSide']

    clean_df['emptyGoalNet'] = empty_goal_func(clean_df).astype(int)
    clean_df['isGoalAdvantage'] = goal_situation(clean_df)

    clean_df['isGoal'] = clean_df['typeDescKey'].apply(lambda x: 1 if x == 'goal' else 0)

    clean_df = zoneshoot(clean_df)

    clean_df.drop('situationCode', axis=1, inplace=True)
    return clean_df
