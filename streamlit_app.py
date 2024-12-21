import streamlit as st
import pandas as pd
import numpy as np
import wandb
import requests
from ift6758.ift6758.client.serving_client import ServingClient
from ift6758.ift6758.client.game_client import GameClient

# Define initial states for game data
if "home_team" not in st.session_state:
    st.session_state.home_team = None
    st.session_state.away_team = None
    st.session_state.period = None
    st.session_state.period_time_remaining = None
    st.session_state.home_score = None
    st.session_state.away_score = None
    st.session_state.home_xg = None
    st.session_state.away_xg = None

ip = "127.0.0.1"
serving_client = ServingClient(ip)
game_client = GameClient(ip)

st.title("Hockey Visualization App")

# Sidebar Inputs (model configuration)
with st.sidebar:
    st.header("Model Configuration")

    # Define available options for the dropdown menus
    available_workspaces = ["philippe-bergeron-7-universit-de-montr-al-org/wandb-registry-model"]
    available_models = ["Logistic regression"]
    available_versions = ["v4 (distance)", "v5 (distance + angle)"]

    # Dropdown menus
    workspace = st.selectbox("Workspace", available_workspaces)
    model_name = st.selectbox("Model", available_models)
    model_version = st.selectbox("Version", available_versions)

    if st.button("Get Model"):
        if workspace and model_name and model_version:
            response = serving_client.download_registry_model(workspace, model_name, model_version[1])
            print(response)
            try:
                # Process the response
                if response.get("status") == "success":
                    st.success(response.get("message", "Model downloaded successfully!"))
                else:
                    st.error(response.get("message", "Failed to download model."))

            except Exception as e:
                st.error(f"Error downloading model: {e}")
        else:
            st.warning("Please specify workspace, model and version.")

# Game ID input
with st.container():
    game_id = st.text_input("Game ID", "")

    if st.button("Ping Game"):
        if game_id:
            try:
                # Query NHL API for game information
                nhl_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
                response = requests.get(nhl_url)
                if response.status_code == 200:
                    game_data = response.json()

                    # Update session state with game data
                    st.session_state.home_team = game_data['homeTeam']['commonName']['default']
                    st.session_state.away_team = game_data['awayTeam']['commonName']['default']
                    st.session_state.period = game_data['periodDescriptor']['number']
                    st.session_state.period_time_remaining = game_data['clock']['timeRemaining']
                    st.session_state.home_score = game_data['homeTeam']['score']
                    st.session_state.away_score = game_data['awayTeam']['score']
                    
                    # Get xG data
                    st.session_state.home_xg = 3.2 
                    st.session_state.away_xg = 1.4

                    st.success("Game data fetched successfully!")
                else:
                    st.error("Invalid Game ID or API error.")
            except Exception as e:
                st.error(f"Error fetching game data: {e}")
        else:
            st.warning("Please enter a Game ID.")

# Game info and predictions
with st.container():    
    # Check if game data exists in the session state
    if st.session_state.home_team and st.session_state.away_team:

        st.subheader(f"Game {game_id}: {st.session_state.home_team} vs {st.session_state.away_team}")

        # Display period and time remaining
        st.markdown(f"**Period {st.session_state.period} - {st.session_state.period_time_remaining} left**")

        # Columns for displaying information side by side
        col1, col2 = st.columns(2)

        # Home team details
        with col1:
            st.subheader(f"{st.session_state.home_team} xG (Actual)")
            st.write(f"**{st.session_state.home_xg} ({st.session_state.home_score})**")
            st.write(f"Difference: {st.session_state.home_xg - st.session_state.home_score:.1f}")

        # Away team details
        with col2:
            st.subheader(f"{st.session_state.away_team} xG (Actual)")
            st.write(f"**{st.session_state.away_xg} ({st.session_state.away_score})**")
            st.write(f"Difference: {st.session_state.away_xg - st.session_state.away_score:.1f}")

        # Game infos
        st.subheader(f"Data used for predictions (and predictions)")
        game_data = game_client.get_game_and_filter(game_id)

        # Predictions
        if model_version == "v4 (distance)":
            features = game_data[['shotDistance']].rename(columns={'shotDistance': 'distance'})
        else:
            features = game_data[['shotDistance', 'shotAngle']].rename(columns={
                'shotDistance': 'distance',
                'shotAngle': 'angle'
            })

        # Call the predict method with renamed features
        predictions = serving_client.predict(features)
        print(predictions)

        # Add predictions back to game_data
        game_data['predictions'] = predictions
        st.dataframe(game_data)

    else:
        st.info("Enter a Game ID and click 'Ping Game' to view game details.")
    

with st.container():
    # TODO: Add data used for predictions
    pass