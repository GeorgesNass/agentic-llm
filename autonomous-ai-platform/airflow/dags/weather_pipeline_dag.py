'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.1.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Weather DAG with full pipeline: data ingestion, transformation, model training, scoring, plotting and cleanup."
'''

## Task (1) fetch_weather_data()
##          Fetch weather data from OpenWeatherMap API & store JSON in /app/raw_files
## Task (2) transform_data_into_csv(n_files=20)
##          Transform 20 latest JSON files into data.csv in /app/clean_data
## Task (3) transform_data_into_csv(n_files=None)
##          Transform all JSON files into fulldata.csv in /app/clean_data
## Task (4') train_model('lr')
##           Train Linear Regression model
## Task (4'') train_model('dt')
##            Train Decision Tree model
## Task (4''') train_model('rf')
##           Train Random Forest model
## Task (5) select_and_save_best_model()
##          Select best model based on validation score and save it

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from airflow.operators.empty import EmptyOperator

import requests
import json
import os
import pandas as pd
import logging
from time import sleep
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
#import matplotlib.pyplot as plt

## NOT REQUIRED BY EXAM : Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## NOT REQUIRED BY EXAM : Detect context execution, Docker vs local
IN_DOCKER = os.getenv("AIRFLOW__CORE__EXECUTOR") is not None

## NOT REQUIRED BY EXAM : Define base directory:
##      - If Docker: /app
##      - Else: go one level up from current file (__file__) to simulate /app/
BASE_DIR = '/app' if IN_DOCKER else Path(__file__).resolve().parents[1]
ROOT_DIR = os.path.dirname(BASE_DIR)
dotenv_path = os.path.join(BASE_DIR, '.env')
print(f"Looking for .env in: {dotenv_path}")

## Define data folders relative to BASE_DIR
RAW_FOLDER = os.path.join(BASE_DIR, 'raw_files')
CLEAN_FOLDER = os.path.join(BASE_DIR, 'clean_data')
SCORES_FILE = os.path.join(CLEAN_FOLDER, 'model_scores.csv')

## Load .env file if present
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

# Parcourir toutes les variables d’environnement actuelles
print("=== Variables d'environnement chargées depuis .env ===")
for key, value in os.environ.items():
    if key in ["WEATHER_API_KEY", "CITIES", "AIRFLOW_UID", "AIRFLOW_GID"]:  # filtre optionnel
        print(f"{key} = {value}")

## API Key
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set WEATHER_API_KEY in your environment or .env file.")

## CITIES
CITIES_RAW = os.getenv("CITIES")
if not CITIES_RAW:
    raise ValueError("CITIES not found in environment or .env file.")
try:
    CITIES = json.loads(CITIES_RAW)
except Exception as e:
    raise ValueError(f"Invalid JSON in CITIES variable: {e}")
## Default parameters for DAG tasks:
##      - 'owner': name of the DAG owner (useful for monitoring/logging)
##      - 'retries': number of retry attempts if a task fails
default_args = {
    'owner': 'airflow',
    'retries': 1
}

## Force extended data generation for training purposes
FORCE_MORE_DATA = True

def fetch_weather_data(cities=None):
    """
        Fetches weather data for each city defined in the CITIES list using the OpenWeatherMap API

        For each city:
            - Sends a GET request with metric units and the provided API key
            - Handles common response codes (200, 401, 429, others)
            - If rate limit (429) is hit, the function waits 10 seconds and retries once
            - All successful responses are saved in a timestamped JSON file under RAW_FOLDER

        No file is created if no valid data is collected
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results = []

    ## Iterate over each city and send a request to the API
    cities = cities or CITIES
    for city in cities:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={API_KEY}&units=metric"
        )

        try:
            res = requests.get(url, timeout=10)

            ## If response is OK, add to result list
            if res.status_code == 200:
                results.append(res.json())

            elif res.status_code == 429:
                logger.warning(f"Rate limit hit for '{city}', waiting 10s before retrying...")
                sleep(10)
                retry = requests.get(url, timeout=10)
                if retry.status_code == 200:
                    results.append(retry.json())
                    logger.info(f"Successfully retried for '{city}'")
                else:
                    logger.warning(f"Retry failed for '{city}' – status code: {retry.status_code}")

            elif res.status_code == 401:
                logger.warning(f"Unauthorized access for city '{city}' (check API key).")

            else:
                logger.warning(f"Failed to fetch data for '{city}' – status code: {res.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network or request error for '{city}': {e}")

    ## Skip file creation if no data was collected
    if not results:
        logger.warning("No weather data collected. File not created.")
        return

    ## Save results to a JSON file with timestamp in filename
    path = os.path.join(RAW_FOLDER, f"{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f)

    logger.info(f"Saved weather data for {len(results)} cities to '{path}'")

def transform_data_into_csv(n_files=None, filename='data.csv'):
    """
        Transforms raw JSON weather files into a CSV file

        This function loads the last `n_files` JSON files (or all if None)
            from RAW_FOLDER, extracts temperature, pressure, and city metadata,
            and writes the result into a CSV file in CLEAN_FOLDER

        Parameters:
            n_files (int or None): Number of recent files to transform
                                   If None, transforms all available files
            filename (str): Name of the output CSV file
    """

    ## Get list of files from raw data folder, sorted by newest first
    files = sorted(os.listdir(RAW_FOLDER), reverse=True)

    if not files:
        logger.warning("No JSON files found in raw folder.")
        return

    if n_files:
        files = files[:n_files]

    dfs = []

    ## Process each file and extract relevant weather fields
    for f in files:
        file_path = os.path.join(RAW_FOLDER, f)

        try:
            with open(file_path, "r") as file:
                data_temp = json.load(file)
        except Exception as e:
            logger.warning(f"Error reading file {f}: {e}")
            continue

        ## Each file may contain multiple cities
        for data_city in data_temp:
            try:
                dfs.append({
                    "temperature": data_city["main"]["temp"],
                    "city": data_city["name"],
                    "pression": data_city["main"]["pressure"],
                    "date": f.split(".")[0]
                })
            except KeyError as e:
                logger.warning(f"Missing key {e} in file {f}")

    if not dfs:
        logger.warning("No valid weather data found; CSV not created.")
        return

    ## Create DataFrame and export to CSV
    df = pd.DataFrame(dfs)

    output_path = os.path.join(CLEAN_FOLDER, filename)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved CSV to '{output_path}' with shape {df.shape}")

def prepare_data(path_to_data=os.path.join(CLEAN_FOLDER, "fulldata.csv")):
    """
        Prepares machine learning data from weather CSV by:
            - Sorting by city and date
            - Adding shifted features (temperature at t+1, ..., t+9)
            - Creating target column (temperature at t-1)
            - One-hot encoding the city column
            - Removing rows with missing values due to shifting

        Parameters:
            path_to_data (str): Path to the input CSV file

        Returns:
            X (DataFrame): Feature matrix
            y (Series): Target values
    """

    if not os.path.exists(path_to_data):
        logger.error(f"Data file not found: {path_to_data}")
        raise FileNotFoundError(f"No file at: {path_to_data}")

    df = pd.read_csv(path_to_data)

    ## Sort by city and then by date
    df = df.sort_values(["city", "date"], ascending=True)

    dfs = []

    ## Create features per city
    for city in df["city"].unique():
        df_temp = df[df["city"] == city].copy()

        ## Target: temp at t-1
        df_temp["target"] = df_temp["temperature"].shift(1)

        ## Features: temperature at t+1 to t+9
        for i in range(1, 10):
            df_temp[f"temp_m-{i}"] = df_temp["temperature"].shift(-i)

        ## Remove rows with NaN values due to shifting
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    if not dfs:
        logger.error("No valid data after shifting.")
        raise ValueError("Insufficient data for training after preprocessing.")

    ## Concatenate all cities
    df_final = pd.concat(dfs, axis=0, ignore_index=False)

    ## Drop date column, which is no longer needed
    df_final = df_final.drop(["date"], axis=1)

    ## One-hot encode the city variable
    df_final = pd.get_dummies(df_final)

    ## Extract features and target
    X = df_final.drop(["target"], axis=1)
    y = df_final["target"]

    logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")

    return X, y

def train_model(model_name):
    """
        Trains a classical regression model (LinearRegression, DecisionTree, RandomForest)
            using cross-validation and returns the average validation score (MSE)

        Parameters:
            model_name (str): One of ['lr', 'dt', 'rf']

        Returns:
            float: Mean cross-validated score (neg_mean_squared_error)
    """

    ## Prepare training data
    X, y = prepare_data()

    ## Define available models
    model_dict = {
        "lr": LinearRegression(),
        "dt": DecisionTreeRegressor(),
        "rf": RandomForestRegressor()
    }

    ## Validate model name
    if model_name not in model_dict:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError(f"Model must be one of {list(model_dict.keys())}")

    model = model_dict[model_name]

    ## Perform 3-fold cross-validation with negative MSE
    score = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error").mean()

    logger.info(f"Model '{model_name}' → Mean CV score: {score:.4f}")

    return score

def select_and_save_best_model(ti=None):
    """
        Selects the best classical model from XCom validation scores
        Retrains it on the full dataset and saves it as a .pkl file
        Logs the model name, score, and timestamp in model_scores.csv
    """

    ## Pull scores from XCom
    if ti:
        scores = {
            'lr': ti.xcom_pull(task_ids='training_models.train_lr'),
            'dt': ti.xcom_pull(task_ids='training_models.train_dt'),
            'rf': ti.xcom_pull(task_ids='training_models.train_rf')
        }
    else:
        ## fallback when executing script locally
        scores = {
            'lr': train_model('lr'),
            'dt': train_model('dt'),
            'rf': train_model('rf')
        }

    ## Select model with the best score (higher = better for neg MSE)
    best_model = max(scores, key=scores.get)

    ## Load full dataset
    X, y = prepare_data()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    ## Instantiate and retrain best model
    if best_model == 'lr':
        model = LinearRegression()
    elif best_model == 'dt':
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()

    model.fit(X, y)

    ## Save model
    model_path = f"{CLEAN_FOLDER}/best_model.pkl"
    dump(model, model_path)

    logger.info(f"✅  Saved best model: {best_model} → {model_path}")

    ## Log score to CSV
    with open(SCORES_FILE, "a") as f:
        f.write(f"{now},{best_model},{scores[best_model]:.4f}\n")

def validate_csv_file(path=None):
    """
        Validates the main dataset CSV file by ensuring:
            - It exists and is not empty
            - It contains no NaN values
            - Temperature values are within realistic bounds (–80°C to +60°C)
    """

    path = path or os.path.join(CLEAN_FOLDER, "fulldata.csv")

    ## Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ## Load CSV
    df = pd.read_csv(path)

    ## Check for empty file
    if df.empty:
        raise ValueError("CSV file is empty")

    ## Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("CSV contains missing (NaN) values")

    ## Check temperature range validity
    if df["temperature"].min() < -80 or df["temperature"].max() > 60:
        raise ValueError("Temperature values out of expected range [-80, 60]")

    logger.info(f"CSV validation passed for {path} (rows: {df.shape[0]})")

## NOT REQUIRED BY EXAM :
# def plot_temperature():
#     """
#         Generates a temperature trend plot for each city using the latest cleaned dataset
#         The plot is saved to 'plot.png' in the clean data folder
#     """

#     path = os.path.join(CLEAN_FOLDER, "data.csv")
#     output_path = os.path.join(CLEAN_FOLDER, "plot.png")

#     ## Load data
#     df = pd.read_csv(path)

#     ## Create plot
#     plt.figure(figsize=(10, 6))

#     ## Plot each city separately
#     for city in df["city"].unique():
#         city_df = df[df["city"] == city]
#         plt.plot(city_df["date"], city_df["temperature"], label=city)

#     ## Add plot formatting
#     plt.legend()
#     plt.title("Temperature Trends by City")
#     plt.xlabel("Date")
#     plt.ylabel("Temperature (°C)")
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     ## Save figure
#     plt.savefig(output_path)
#     logger.info(f"Temperature plot saved to {output_path}")

def cleanup_old_files():
    """
        Removes all but the 20 most recent .json and .csv files
        from both the raw and clean data folders.
    """

    for folder in [RAW_FOLDER, CLEAN_FOLDER]:
        ## Select only .json or .csv files, sorted by name descending
        files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".json") or f.endswith(".csv")],
            reverse=True
        )

        ## Keep only the 20 most recent files
        old_files = files[20:]
        for f in old_files:
            os.remove(os.path.join(folder, f))

        logger.info(f"Cleaned {len(old_files)} old file(s) from {folder}")

## Define the DAG configuration and metadata
with DAG(
    dag_id='weather_pipeline_dag',
    description='Fetch, transform, validate, model weather data and update dashboard',
    schedule_interval='* * * * *',  ## DAG runs every minute
    start_date=days_ago(1),         ## Static start date
    catchup=False,                  ## Prevents catching up of missed intervals
    default_args=default_args,      ## Applies owner/retry policy to all tasks
    tags=['weather', 'etl', 'ml', 'datascientest'],  ## Useful tags for categorizing DAGs in UI
) as dag:

    ## Mark the start of the DAG execution
    start = EmptyOperator(task_id='start')

    ## Mark the end of the DAG execution
    end = EmptyOperator(task_id='end')

    ## Step (1): Fetch weather data from OpenWeatherMap API
    ## Saves a JSON file timestamped to the minute in raw_files/
    fetch_data = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data
    )

    ## Step (2): Transform the 20 most recent raw JSON files into a cleaned CSV
    ## Output: clean_data/data.csv (for quick plots)
    transform_recent = PythonOperator(
        task_id='transform_recent_json',
        python_callable=transform_data_into_csv,
        op_kwargs={'n_files': 20, 'filename': 'data.csv'}
    )

    ## Step (3): Transform all raw JSON files into a global CSV
    ## Output: clean_data/fulldata.csv (used for training models)
    transform_all = PythonOperator(
        task_id='transform_all_json',
        python_callable=transform_data_into_csv,
        op_kwargs={'n_files': None, 'filename': 'fulldata.csv'}
    )

    ## Step (4): Validate the full dataset CSV before model training
    ## Ensures no NaNs and realistic temperature ranges
    validate_fulldata = PythonOperator(
        task_id='validate_csv_file',
        python_callable=validate_csv_file
    )

    ## Step (5): Generate and save a line plot of temperatures per city
    ## Output: clean_data/plot.png
    # plot_temp = PythonOperator(
    #     task_id='plot_temperature',
    #     python_callable=plot_temperature
    # )

    ## Step (6): Train 3 ML models in parallel using a TaskGroup
    ## Includes: Linear Regression, Decision Tree, Random Forest
    with TaskGroup('training_models') as training_group:
        train_lr = PythonOperator(task_id='train_lr', python_callable=train_model, op_args=['lr'])
        train_dt = PythonOperator(task_id='train_dt', python_callable=train_model, op_args=['dt'])
        train_rf = PythonOperator(task_id='train_rf', python_callable=train_model, op_args=['rf'])
        # train_lstm = PythonOperator(...) was removed

    ## Step (7): Select the best model using validation scores
    ## Retrains the selected model on all data and saves it
    select_best_model = PythonOperator(
        task_id='select_and_save_best_model',
        python_callable=select_and_save_best_model
    )

    ## Step (8): Cleanup older files (keep only the 20 latest in each folder)
    cleanup_task = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=cleanup_old_files
    )

    ## NOT REQUIRED BY EXAM : Optional sensor
    ##      wait until a dummy file is available (simulate async readiness)
    wait_for_file = FileSensor(
        task_id='wait_for_raw_data',
        filepath=f"{RAW_FOLDER}/null_file.json",
        poke_interval=30,  ## check every 30 seconds
        timeout=120,       ## wait for a max of 2 minutes
        mode="poke"
    )

    ## Define DAG execution flow (task dependencies)
    start >> fetch_data >> wait_for_file >> transform_recent >> transform_all
    transform_all >> validate_fulldata >> training_group >> select_best_model
    #select_best_model >> [plot_temp, cleanup_task] >> end
    select_best_model >> [cleanup_task] >> end

## ===============================================================
## OPTIONAL: Local execution entry point to simulate the DAG logic
## ===============================================================
if __name__ == "__main__":

    ## Ensure required directories exist or ask user
    for folder_var, folder_path in [("RAW_FOLDER", RAW_FOLDER), ("CLEAN_FOLDER", CLEAN_FOLDER)]:
        if not os.path.exists(folder_path):
            print(f"\n{folder_var} not found at: {folder_path}")
            user_input = input(f"Enter a custom path to create {folder_var} (or press Enter to use default): ").strip()
            if user_input:
                folder_path = os.path.abspath(user_input)
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            print(f"{folder_var} created at: {folder_path}")

            ## Update the path in global variable
            if folder_var == "RAW_FOLDER":
                RAW_FOLDER = folder_path
            else:
                CLEAN_FOLDER = folder_path

    ## Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LOCAL_RUN")

    ## Step 1: Fetch weather data (more times if FORCE_MORE_DATA is True)
    ## NOT REQUIRED BY EXAM
    ##     we do this in order not to have the error
    ##          ERROR - Model training or selection failed: Cannot have number of splits n_splits=3 greater than the number of samples: n_samples=0.
    ##     because of insuffisient data for training
    ## NOT REQUIRED BY EXAM
    try:
        n_iterations = 30 if FORCE_MORE_DATA else 1
        for i in range(n_iterations):
            logger.info(f"Fetching weather data – iteration {i+1}/{n_iterations}")
            fetch_weather_data(cities=CITIES)
            sleep(60)  ## slight delay between fetches
        logger.info("Weather data fetched.")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")

    ## Step 2: Transform recent JSON files
    try:
        transform_data_into_csv(n_files=20, filename="data.csv")
        logger.info("Recent JSON transformed into data.csv.")
    except Exception as e:
        logger.error(f"Error transforming recent data: {e}")

    ## Step 3: Transform all JSON files
    try:
        transform_data_into_csv(n_files=None, filename="fulldata.csv")
        logger.info("All JSON transformed into fulldata.csv.")
    except Exception as e:
        logger.error(f"Error transforming full data: {e}")

    ## Step 4: Validate CSVs
    try:
        validate_csv_file(os.path.join(CLEAN_FOLDER, "data.csv"))
        validate_csv_file(os.path.join(CLEAN_FOLDER, "fulldata.csv"))
        logger.info("CSV files validated.")
    except Exception as e:
        logger.warning(f"CSV validation failed: {e}")

    ## Step 5: Train and save best model
    try:

        X, y = prepare_data(path_to_data=os.path.join(CLEAN_FOLDER, "fulldata.csv"))
        scores = {}
        for model_type in ["lr", "dt", "rf"]:
            score = train_model(model_name=model_type)
            scores[model_type] = score
            logger.info(f"{model_type} score: {score:.3f}")

        select_and_save_best_model(ti=None)
        logger.info("Best model selected and saved.")
    except Exception as e:
        logger.error(f"Model training or selection failed: {e}")

    ## NOT REQUIRED BY EXAM : Step 6 Plotting
    #user_plot = input("\nDo you want to generate a temperature plot? (y/n): ").strip().lower()
    #if user_plot == "y":
    #    try:
    #        plot_temperature()
    #        logger.info("Temperature plot generated.")
    #    except Exception as e:
    #        logger.warning(f"Plotting failed: {e}")
    #else:
    #    logger.info("Skipped plotting.")

    ## NOT REQUIRED BY EXAM : Step 7 Cleanup
    user_cleanup = input("Do you want to delete old files? (y/n): ").strip().lower()
    if user_cleanup == "y":
        try:
            cleanup_old_files()
            logger.info("Old files cleaned.")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    else:
        logger.info("Skipped cleanup.")
