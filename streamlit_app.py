import streamlit as st
import pandas as pd
import requests
from ift6758.ift6758.client.serving_client import ServingClient
from ift6758.ift6758.client.game_client import GameClient

# Utility function to reset session state
def reset_session_state():
    session_keys = [
        "home_team", "away_team", "period", "period_time_remaining",
        "home_score", "away_score", "home_xg", "away_xg",
        "play_by_play", "last_ping_id"
    ]
    for key in session_keys:
        st.session_state[key] = None

# Utility function to reset session state for a specific game ID
def reset_game_state(game_id):
    st.session_state.play_by_play_data[game_id] = pd.DataFrame()

# Initialize dictionary for storing play_by_play data for each game
if "play_by_play_data" not in st.session_state:
    st.session_state.play_by_play_data = {}

# Initialize session state variables
if "home_team" not in st.session_state:
    reset_session_state()
    st.session_state.ping_game_clicked = False

# Set up IP address for clients
ip = "127.0.0.1"

# Initialize ServingClient
if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip)
serving_client = st.session_state.serving_client

# Initialize GameClient
if "game_client" not in st.session_state:
    st.session_state.game_client = GameClient(ip)
game_client = st.session_state.game_client

# App title
st.title("Hockey Visualization App")

# Sidebar: Model Configuration
with st.sidebar:
    st.header("Model Configuration")
    
    available_workspaces = ["philippe-bergeron-7-universit-de-montr-al-org/wandb-registry-model"]
    available_models = ["Logistic regression"]
    available_versions = ["v6 (distance)", "v5 (distance + angle)"]

    workspace = st.selectbox("Workspace", available_workspaces)
    model_name = st.selectbox("Model", available_models)
    model_version = st.selectbox("Version", available_versions)

    if st.button("Get Model"):
        if workspace and model_name and model_version:
            try:
                response = serving_client.download_registry_model(workspace, model_name, model_version[1])
                if response.get("status") == "success":
                    st.success(response.get("message", "Model downloaded successfully!"))
                else:
                    st.error(response.get("message", "Failed to download model."))
            except Exception as e:
                st.error(f"Error downloading model: {e}")
        else:
            st.warning("Please specify workspace, model, and version.")

# Game ID input and logic
with st.container():
    game_id = st.text_input("Game ID", "")
    
    if st.button("Ping Game"):
        st.session_state.ping_game_clicked = True
        if game_id:
            if game_id not in st.session_state.play_by_play_data:
                reset_game_state(game_id)  # Initialize play_by_play data for new game IDs
            
            # Update the last pinged game ID
            st.session_state.last_ping_id = game_id

            try:
                nhl_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
                response = requests.get(nhl_url)
                if response.status_code == 200:
                    game_data = response.json()
                    
                    # Store game-specific data in session state
                    st.session_state.home_team = game_data['homeTeam']['commonName']['default']
                    st.session_state.away_team = game_data['awayTeam']['commonName']['default']
                    st.session_state.period = game_data['periodDescriptor']['number']
                    st.session_state.period_time_remaining = game_data['clock']['timeRemaining']
                    st.session_state.home_score = game_data['homeTeam']['score']
                    st.session_state.away_score = game_data['awayTeam']['score']
                    
                    # Placeholder xG values
                    st.session_state.home_xg = 3.2
                    st.session_state.away_xg = 1.4

                    st.success("Game data fetched successfully!")
                else:
                    st.error("Invalid Game ID or API error.")
            except Exception as e:
                st.error(f"Error fetching game data: {e}")
        else:
            st.warning("Please enter a Game ID.")

# Display game info and predictions
with st.container():
    if st.session_state.last_ping_id:
        current_game_id = st.session_state.last_ping_id
        st.subheader(f"Game {current_game_id}: {st.session_state.home_team} vs {st.session_state.away_team}")
        st.markdown(f"**Period {st.session_state.period} - {st.session_state.period_time_remaining} left**")
        
        col1, col2 = st.columns(2)

        # Home team details
        with col1:
            st.subheader(f"{st.session_state.home_team} xG (Actual)")
            st.write(f"**{st.session_state.home_xg} ({st.session_state.home_score})**")
            diff_home = st.session_state.home_xg - st.session_state.home_score
            arrow_home = "↑" if diff_home > 0 else "↓"
            color_home = "green" if diff_home > 0 else "red"
            st.markdown(f"<span style='color:{color_home}; font-size:1.5em;'>{arrow_home}</span> **{abs(diff_home):.1f}**", unsafe_allow_html=True)

        # Away team details
        with col2:
            st.subheader(f"{st.session_state.away_team} xG (Actual)")
            st.write(f"**{st.session_state.away_xg} ({st.session_state.away_score})**")
            diff_away = st.session_state.away_xg - st.session_state.away_score
            arrow_away = "↑" if diff_away > 0 else "↓"
            color_away = "green" if diff_away > 0 else "red"
            st.markdown(f"<span style='color:{color_away}; font-size:1.5em;'>{arrow_away}</span> **{abs(diff_away):.1f}**", unsafe_allow_html=True)

        # Fetch play-by-play probabilities
        new_plays = game_client.get_game_and_filter(current_game_id)
        if new_plays is not None:
            features = ['shotDistance'] if model_version == "v6 (distance)" else ['shotDistance', 'shotAngle']
            serving_client.features = features

            pred_and_prob = serving_client.predict(new_plays)
            new_plays['probabilities'] = pred_and_prob['probabilities']

            # Update play_by_play dataframe for the current game
            st.session_state.play_by_play_data[current_game_id] = pd.concat(
                [st.session_state.play_by_play_data[current_game_id], new_plays], ignore_index=True
            )

        # Display play_by_play data for the current game
        st.dataframe(st.session_state.play_by_play_data[current_game_id])
    else:
        st.info("Enter a Game ID and click 'Ping Game' to view game details.")