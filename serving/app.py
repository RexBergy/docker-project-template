"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, json, jsonify, request, abort
import requests
import pandas as pd
import joblib
import wandb


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_DIR = Path("models")


MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    logging.info("App initialized")

    global current_model
    global current_model_name

    try:
        model_file_path = MODEL_DIR / "logistic_regression_distance1.pkl"
        if model_file_path.exists():
            logging.info(model_file_path)
            current_model = joblib.load(model_file_path)
            current_model_name = "default_model"
            logging.info("Loaded default model: default_model")
        else:

            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb.init()
            model_artifact = wandb.use_artifact(f"philippe-bergeron-7-universit-de-montr-al-org/wandb-registry-model/Logistic regression:v6")
            model_artifact.download(root=str(MODEL_DIR))
            wandb.finish()

            # Load the downloaded model
            
            logging.info(model_file_path)
            current_model = joblib.load(model_file_path)
            current_model_name = f"Default model (Logistic Regression Distance)"
            app.logger.info(f"Downloaded and loaded model: {current_model_name}")
            response = {"status": "success", "message": f"Downloaded and loaded model {current_model_name}"}
            return response
    except Exception as e:
        logging.error(f"Failed to load default model: {e}")

before_first_request()

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    try:
        with open(LOG_FILE, "r") as f:
            log_data = f.read()
        return jsonify({"logs": log_data})
    except Exception as e:
        logging.error(f"Error reading logs: {e}")
        abort(500, "Failed to read logs")
#     # TODO: read the log file specified and return the data
#     #raise NotImplementedError("TODO: implement this endpoint")

#     #response = None
#     #return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    #Get POST json data
    global current_model
    global current_model_name

    json_request = request.get_json()
    app.logger.info(f"Received request to download model: {json_request}")

    data = json.loads(json_request)

    workspace = data.get("workspace")
    model_name = data.get("model")
    version = data.get("version")
    wandb.init()
    model_artifact = wandb.use_artifact(f"{workspace}/{model_name}:v{version}")
    model_file_path = Path(model_artifact.file(root=str(MODEL_DIR)))

    try:
        if model_file_path.exists():
            wandb.finish()
            app.logger.info(f"Model already exists: {model_file_path}. Loading model.")
            
            current_model = joblib.load(model_file_path)
            current_model_name = f"{model_name}_v{version}"
            response = {"status": "success", "message": f"Loaded model {current_model_name}"}
        else:
            model_artifact.download(root=str(MODEL_DIR))
            wandb.finish()

            # Load the downloaded model
            current_model = joblib.load(model_file_path)
            current_model_name = f"{model_name}_v{version}"
            app.logger.info(f"Downloaded and loaded model: {current_model_name}")
            response = {"status": "success", "message": f"Downloaded and loaded model {current_model_name}"}
    except Exception as e:
        logging.error(f"Failed to download/load model: {e}")
        response = {"status": "failure", "message": str(e)}



    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_data = request.get_json()
    app.logger.info(f"Received prediction request: {json_data}")
    data = pd.DataFrame(json_data)
    try:
        # Assuming features are in a format suitable for the current model
        predictions = current_model.predict(data)
        probabilities = current_model.predict_proba(data)
        response = {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
        app.logger.info(response)
        return jsonify(response) 
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        response = {"status": "failure", "message": str(e)}
        return response

