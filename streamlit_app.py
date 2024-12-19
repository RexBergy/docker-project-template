import streamlit as st
import pandas as pd
import numpy as np
import wandb
import requests

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

st.title("Hockey Visualization App")

# Sidebar Inputs (model configuration)
with st.sidebar:
    st.header("Model Configuration")

    # Define available options for the dropdown menus
    available_workspaces = ["philippe-bergeron-7-universit-de-montr-al-org"]
    available_models = ["Logistic regression"]
    available_versions = ["v4 (distance)", "v5 (distance + angle)"]

    # Use selectbox for dropdown menus
    workspace = st.selectbox("Workspace", available_workspaces)
    model_name = st.selectbox("Model", available_models)
    model_version = st.selectbox("Version", available_versions)
    project_name = "wandb-registry-model"

    if st.button("Get Model"):
        if workspace and model_name and model_version:
            try:
                # Initialize WandB API client
                api = wandb.Api()

                # Extract the version number from the selected dropdown
                version = model_version.split()[0]

                # Download the model from the registry
                artifact = api.artifact(f"{workspace}/{project_name}/{model_name}:{model_version.split()[0]}")
                artifact_dir = artifact.download()
                st.success(f"Model {model_name}:{version} downloaded successfully!")
                
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
                    
                    # Simulated xG data
                    # TODO: Replace with actual xG data from the game client
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

    else:
        st.info("Enter a Game ID and click 'Ping Game' to view game details.")
    

with st.container():
    # TODO: Add data used for predictions
    pass