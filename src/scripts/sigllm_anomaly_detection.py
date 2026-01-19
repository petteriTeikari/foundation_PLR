# Steps to reproduce:
import datetime
import os

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.scripts.mlflow_utils import init_mlflow, init_mlflow_experiment

MLFLOW_TRACKING_URI = "repo_desktop_clone/foundation_PLR/src/mlruns"
MLFLOW_EXPERIMENT_NAME = "PLR_OutlierDetection"
MLFLOW_RUN_BASE_NAME = "SigLLM"

# SigLLM uses OpenAI API, so you need to have your OpenAI account and have the secret key in your environment
# e.g, in your .bashrc (gedit and add to the end of the file, or use .env file with dotenv package):
# https://github.com/sintel-dev/sigllm/issues/24#issuecomment-2447987147
# export OPENAI_API_KEY=[your secret key]
# source ~/.bashrc
# See that it is there with "printenv" command

# 1) Clone the repo:
# git clone https://github.com/petteriTeikari/sigllm.git

# 2) Install the dependencies: (with setup.py)
# virtualenv -p /usr/bin/python3.8 .venv-sigllm
# source .venv-sigllm/bin/activate
# pip install .
# pip install mlflow

# 3) Use the created venv to run this script then


def get_repo_root(base_name: str = "repo_PLR"):
    cwd = os.getcwd()
    if os.path.basename(cwd) == "src":
        repo_root = os.path.dirname(cwd)
    elif os.path.basename(cwd) != base_name:
        # subfolder in "src", make recursive later
        init = os.path.dirname(cwd)
        repo_root = os.path.dirname(init)
    else:
        repo_root = cwd
    return repo_root


def get_data_dir(data_path: str = "data"):
    data_dir = os.path.join(get_repo_root(), data_path)
    return data_dir


def create_ts(X_subj, time_interval_s: int, fps: int = 30):
    time_vector = (
        np.linspace(0, X_subj.shape[0] - 1, X_subj.shape[0]) / fps
    )  # in seconds
    a = datetime.datetime(2000, 1, 1, 12, 00, 00)
    if time_interval_s == fps:
        print("Use correct time")
        ds = [
            a + datetime.timedelta(0, t) for t in time_vector
        ]  # add seconds to a dummy date
    else:
        print(f"Use {time_interval_s}-sec interval")
        # Maybe not all methods like the "biosignal sample rate" as most libraries seem to have
        # second the shortest time, if you have some glitches with too dense sampling,
        # try this
        ds = []
        for idx in range(len(time_vector)):
            ds.append(a + datetime.timedelta(0, seconds=idx))
    # this needs to be timestamp, not datetime.datetime for SigLLM
    ds = pd.DataFrame(np.array(ds))
    ts = ds.values.astype(np.int64) // 10**9

    return ts.flatten()


def create_timeseries_df(X, time_interval_s: int):
    ts = create_ts(X, time_interval_s=time_interval_s)
    df = pd.DataFrame({"timestamp": ts, "value": X.flatten()})
    return df


def return_pred_mask(anomalies, df, X_subj):
    """
    anomalies: df: pd.DataFrame
       will contain rows with the timestamps of the detected anomalies with the severity score
       https://github.com/petteriTeikari/sigllm?tab=readme-ov-file#detect-anomalies-using-a-sigllm-pipeline
    """
    df["anomaly"] = False
    df["severity"] = np.nan
    for i, row in anomalies.iterrows():
        # get the timestamp of the anomaly
        start, end = row["start"], row["end"]
        # find the index of the timestamp in the original time series
        bool_series = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        severity = row["severity"]
        df["anomaly"] = bool_series
        df.loc[bool_series, "severity"] = severity

    pred_mask = df["anomaly"].to_numpy().astype(int)
    severity = df["severity"].to_numpy().astype(float)

    return pred_mask, severity


def demo_data(sig_llm):
    """
    Test that OpenAI works
    """
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "..", "tutorials", "data.csv"))
    if os.path.exists(abs_path):
        df = pd.read_csv(abs_path)  # e.g. (200, 2)
        # dates = pd.to_datetime(df["timestamp"], unit="s")
        # 2008-10-01 06:00:00 -> 2008-11-20 00:00:00 (6h interval)
        _ = sig_llm.detect(df)
        return None
    else:
        raise FileNotFoundError("Cannot find demo data file = {}".format(abs_path))


def sigllm_subjectwise_detection(
    sig_llm,
    subj_idx: int,
    X_subj: np.ndarray,
    y_subj: np.ndarray,
    time_interval_s: int,
    use_demo_data: bool = False,
    test_with_subset: bool = False,
    split: str = "train",
    print_debug: bool = True,
    train_method: str = "zeroshot",
):
    # https://github.com/petteriTeikari/sigllm?tab=readme-ov-file#detect-anomalies-using-a-sigllm-pipeline
    if use_demo_data:
        demo_data(sig_llm)
        return None
    else:
        df = create_timeseries_df(
            X_subj, time_interval_s=time_interval_s
        )  # e.g. shape (1981,2)
        start_time = datetime.datetime.now()
        if test_with_subset:
            print("test with subset")
            df = df.iloc[0:200, :]
            y_subj = y_subj[0:200]

        if split == "train":
            if train_method == "zeroshot":
                # (13 Nov 2024): $9.80 -> $7.68 = costs $2.12 to detect outliers per subject
                # for 1981 sample timeseries
                #   2%|▏         | 37/1841 [00:39<32:22,  1.08s/it]
                anomalies = sig_llm.detect(df)
            elif train_method == "finetune":
                raise NotImplementedError("Finetune not implemented yet")
                # TODO! make df contain all the data points of the train split
                # Univariate multiple time series fit()
                # https://github.com/sintel-dev/Orion/blob/master/tutorials/Orion_on_Custom_Data.ipynb
                # https://github.com/sintel-dev/Orion/blob/master/tutorials/Orion_with_Multivariate_Input.ipynb
                # anomalies = sig_llm.detect(df, y=y_subj)
        else:
            anomalies = sig_llm.detect(df)

        pred_mask, severity = return_pred_mask(anomalies, df, X_subj)
        time_elapsed = (datetime.datetime.now() - start_time).total_seconds()

        if print_debug:
            print(f"Number of outliers (gt) = {np.sum(y_subj)}")
            print(f"Outlier detection took  = {time_elapsed} seconds")
            print(f"Number of outliers detected = {np.sum(pred_mask)}")
            # df_out = df

    return pred_mask, severity, time_elapsed


def sigllm_splitwise_detection(
    sig_llm,
    X: np.ndarray,
    y: np.ndarray,
    split: str,
    train_method: str,
    time_interval_s: int,
):
    print(f"Splitwise detection: {split}")
    no_of_subjects = X.shape[0]
    pred_masks = np.zeros((no_of_subjects, X.shape[1]))
    severities = np.zeros((no_of_subjects, X.shape[1]))
    times = []
    for subj_idx in tqdm(
        range(no_of_subjects), "Getting anomalies with SigLLM (OpenAI API)"
    ):
        pred_mask, severity, time_elapsed = sigllm_subjectwise_detection(
            sig_llm,
            subj_idx,
            X_subj=X[subj_idx, :],
            y_subj=y[subj_idx, :],
            split=split,
            train_method=train_method,
            time_interval_s=time_interval_s,
        )
        pred_masks[subj_idx, :] = pred_mask
        severities[subj_idx, :] = severity
        times.append(time_elapsed)

    return pred_masks, times, severities


def import_data(train_on: str = "pupil_orig_imputed"):
    # the write here: write_numpy_to_disk() in outlier_sigllm.py
    data_dir = get_data_dir()
    X = np.load(os.path.join(data_dir, f"{train_on}__X.npy"))
    y = np.load(os.path.join(data_dir, f"{train_on}__y.npy"))
    X_test = np.load(os.path.join(data_dir, f"{train_on}__X_test.npy"))
    y_test = np.load(os.path.join(data_dir, f"{train_on}__y_test.npy"))

    return X, y, X_test, y_test


def sigllm_anomaly_detection(
    pipeline_type: str, train_method: str, time_interval_s: int = 1
):
    # MLflow
    if MLFLOW_TRACKING_URI is not None:
        init_mlflow(MLFLOW_TRACKING_URI)
        if MLFLOW_EXPERIMENT_NAME is not None:
            init_mlflow_experiment(MLFLOW_EXPERIMENT_NAME)
            if MLFLOW_RUN_BASE_NAME is not None:
                mlflow_run_name = (
                    MLFLOW_RUN_BASE_NAME + "-" + pipeline_type + "-" + train_method
                )
        else:
            raise ValueError("No mlflow experiment name specified")
    else:
        raise ValueError(
            "No MLflow tracking uri specified, not logging the experiment to MLflow"
        )

    # Import the hacky export from "main script"
    X, y, X_test, y_test = import_data()

    # Init the model
    from sigllm import SigLLM

    # default "interval": 21600sec / (6hours)
    hyperparameters = {
        "orion.primitives.timeseries_anomalies.find_anomalies#1": {
            "fixed_threshold": False
        },
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "interval": time_interval_s  # 1/30 # 1/fps
        },
    }

    with mlflow.start_run(run_name=mlflow_run_name):
        if pipeline_type == "Detection":
            try:
                # you can set the key manually here as well (not recommended)
                os.environ["OPENAI_API_KEY"] = "secret_key"
                sig_llm = SigLLM(
                    pipeline="gpt_detector", decimal=6, hyperparameters=hyperparameters
                )
            except Exception as e:
                # e.g. openai.OpenAIError: The api_key client option must be set either by passing api_key to the
                #                          client or by setting the OPENAI_API_KEY environment variable
                print(f"Error: {e}")
                raise e
        elif pipeline_type == "Prompter":
            raise NotImplementedError("Prompter pipeline not yet implemented")
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        mlflow.log_param("pipeline_type", pipeline_type)

        pred_masks = {}
        split = "train"
        pred_masks[split], train_times, train_severities = sigllm_splitwise_detection(
            sig_llm,
            X=X,
            y=y,
            split=split,
            train_method=train_method,
            time_interval_s=time_interval_s,
        )

        split = "test"
        pred_masks[split], test_times, test_severities = sigllm_splitwise_detection(
            sig_llm,
            X=X_test,
            y=y_test,
            split=split,
            train_method=train_method,
            time_interval_s=time_interval_s,
        )

        # TODO! MLflow log


if __name__ == "__main__":
    train_method = "zeroshot"
    pipeline_types = ["Detection"]  # , "Prompter"]
    for pipeline_type in pipeline_types:
        print(f"Running '{pipeline_type}' pipeline")
        sigllm_anomaly_detection(pipeline_type, train_method)
