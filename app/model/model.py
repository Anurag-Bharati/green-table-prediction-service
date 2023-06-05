import pickle
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

PARAMS = ["id", "week", "meal_id", "checkout_price", "base_price",
          "emailer_for_promotion", "homepage_featured"]


# Load the XGBoost model from the pickle file
def load_model(filename):
    """
    Load the XGBoost model from the pickle file
    :param filename: name of the ml model
    :return: Model or None
    """
    try:
        with open(filename, 'rb') as file:
            xgb_model = pickle.load(file)
        return xgb_model
    except (IOError, ValueError):
        return None


# Load the XGBoost model
model_filename = f'{BASE_DIR}/trained-model-{__version__}.pkl'
model = load_model(model_filename)


# Make predictions using the loaded model
def predict_orders(data):
    """
    Make predictions using the loaded model
    :param data: 2D list of Params
    :return: 1D list of no. of orders
    """
    try:
        return model.predict(data)
    except (ValueError, RuntimeError, Exception):
        return None

