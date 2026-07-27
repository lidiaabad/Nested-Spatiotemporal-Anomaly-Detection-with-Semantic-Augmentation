import numpy as np
import pandas as pd
from pathlib import Path
import torch, random, time, os
from config.reporter import Reporter
from statsmodels.tsa.stattools import acf, pacf
#from torchinfo  import summary

#File with several helpers function


def setup_device(args):
    "Sets the random seed and selects the computation device (CPU or GPU)."
    set_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        print("Running on GPU")
        return torch.device("cuda")
    print("Running on CPU")
    return torch.device("cpu")

def set_seed(seed):
    "Sets random seeds for PyTorch, NumPy, and Python to ensure reproducibility."
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    
def build_experiment_path(args, entity_id=None):
    "Creates and returns the directory where experiment outputs will be stored."
    base = f"{args.model_id}_{args.dataset}"
    if entity_id is not None:
        base += f"_{entity_id}"

    experiment_path = os.path.join(args.checkpoint, base)
    os.makedirs(experiment_path, exist_ok=True)
    return experiment_path

def build_loaded_path(args):
    "Builds the path to a pretrained model checkpoint."
    loaded_dir = os.path.join(args.checkpoint, f"{args.loaded_path}_{args.dataset}")
    os.makedirs(loaded_dir, exist_ok=True)

    # full path to the .pt file
    pretext_path = os.path.join(loaded_dir, f"{args.loaded_path}")
    return pretext_path

def setup_experiment(args, entity_ctx, device):
    "Initializes the experiment context, reporter, and computation device."
    reporter = Reporter(entity_ctx["experiment_path"],f"{args.model_id}_report.json")
    reporter.report("arguments", vars(args))
    reporter.report("experiment_date", entity_ctx["time"])

    entity_ctx["reporter"] = reporter
    entity_ctx["device"] = device
    
    return entity_ctx

def setup_experiment_ssl(args, entity_ctx, device):
    
    pretext_path = os.path.join(entity_ctx["experiment_path"], "pretext")
    reporter_pretext = Reporter(pretext_path,f"{args.model_id}_report.json")
    reporter_pretext.report("arguments", vars(args))
    reporter_pretext.report("experiment_date", entity_ctx["time"])
    
    downstream_path = os.path.join(entity_ctx["experiment_path"], "downstream")
    reporter_downstream = Reporter(downstream_path,f"{args.model_id}_report.json")
    reporter_downstream.report("arguments", vars(args))
    reporter_downstream.report("experiment_date", entity_ctx["time"])

    entity_ctx["reporter_pretext"] = reporter_pretext
    entity_ctx["reporter_downstream"] = reporter_downstream
    entity_ctx["device"] = device
    
    return entity_ctx

def get_weights(data_path):
    '''
    Computes feature weights from normalized feature variances in the training data.
    '''
    
    full_data = pd.read_csv(data_path).iloc[:, 1:] #eliminate datetime
    
    scaled_data = (full_data - full_data.min()) / (full_data.max() - full_data.min() + 1e-8)
    scaled_data = scaled_data.fillna(0)

    # Compute variance of each scaled feature
    vars = scaled_data.var()
    print("vars", vars)
    weights = 0.5 + vars * (1.0 - 0.5) #scale: clip [0.5-1]
    
    return torch.tensor(weights.values, dtype=torch.float32)

def get_weights_public(train_x, test_x):
    '''
    Computes feature weights based on train-test distribution shifts using Z-score statistics.'''

    # Ensure DataFrame format
    if isinstance(train_x, np.ndarray):
        train_x = pd.DataFrame(train_x)
    if isinstance(test_x, np.ndarray):
        test_x = pd.DataFrame(test_x)

    # Compute mean and std on train, then compare with test
    mean_train = train_x.mean()
    mean_test = test_x.mean()
    std_train = train_x.std() + 1e-6  # prevent division by zero

    # Z-score shift and std ratio
    mean_shift = np.abs(mean_test - mean_train) / std_train
    weights=1/((mean_shift+1e-6)**2)
    weights=np.log1p(weights) #smooth
    weights = weights * len(weights) / np.sum(weights) #normalize
    #weights=np.sqrt(weights)
    return torch.tensor(weights.values, dtype=torch.float32)

def get_forward_time(model, device):
    " Measures the execution time of a single forward pass through the model."
    model.to('cpu')
    dummy_data = torch.rand(1, 24000)
    start = time.time()
    _ = model(dummy_data)
    end = time.time()
    forward_time = end - start
    model.to(device)
    return forward_time

def remove_folder(path):
    "Deletes a directory and all of its contents."
    folder = Path(path)
    if folder.exists() and folder.is_dir():
        for item in folder.rglob('*'):
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        folder.rmdir()
    
def time_series_analysis(raw_df, series=None):
    '''
    Summarizes the temporal characteristics of multivariate time series using ACF/PACF analysis.
    '''
    start_time=time.time()
    if series is None:
        series = raw_df
        
    #compute points per hour
    raw_df = raw_df.sort_index()
    deltas = raw_df.index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = deltas.median()
    hour_freq = int(round(3600 / median_seconds))
    
    lag_sets = {
        "hour": hour_freq,
        "day": hour_freq * 24,
        "week": hour_freq * 24 * 7,
    }

    # Contadores globales
    count_ar = 0
    count_ma = 0
    count_arma = 0
    count_noise = 0
    count_season_hour = 0
    count_season_day = 0
    count_season_week = 0

    N = len(series.columns)

    for col in series.columns:
        x = series[col].dropna().values

        # ACF/PACF
        try:
            acf_vals = acf(x, nlags=lag_sets["week"])
            pacf_vals = pacf(x, nlags=lag_sets["week"])
        except:
            continue

        # Threshold
        conf = 1.96 / np.sqrt(len(x))

        acf_sig = np.where(np.abs(acf_vals) > conf)[0]
        pacf_sig = np.where(np.abs(pacf_vals) > conf)[0]

        acf_sig = acf_sig[acf_sig > 0]
        pacf_sig = pacf_sig[pacf_sig > 0]

        # Detect AR / MA / ARMA
        is_ar = len(pacf_sig) > 0 and pacf_sig[0] < 5 and len(acf_sig) > 5
        is_ma = len(acf_sig) > 0 and acf_sig[0] < 5 and len(pacf_sig) > 5
        is_arma = len(acf_sig) > 5 and len(pacf_sig) > 5
        is_noise = len(acf_sig) == 0 and len(pacf_sig) == 0

        if is_ar: count_ar += 1
        if is_ma: count_ma += 1
        if is_arma: count_arma += 1
        if is_noise: count_noise += 1

        # Seasonality
        if abs(acf_vals[lag_sets["hour"]]) > conf:
            count_season_hour += 1
        if abs(acf_vals[lag_sets["day"]]) > conf:
            count_season_day += 1
        if abs(acf_vals[lag_sets["week"]]) > conf:
            count_season_week += 1

    # ---------------------------------------------------------
    # RESUMEN GLOBAL
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("GLOBAL TEMPORAL SUMMARY")
    print("="*80)
    print(f"Variables totales: {N}")
    print(f"\nAR structure: {count_ar}/{N} variables")
    print(f"MA structure: {count_ma}/{N} variables")
    print(f"ARMA structure: {count_arma}/{N} variables")
    print(f"Noise-like: {count_noise}/{N} variables")
    print("\nSeasonality:")
    print(f"  Hourly:  {count_season_hour}/{N}")
    print(f"  Daily:   {count_season_day}/{N}")
    print(f"  Weekly:  {count_season_week}/{N}")
    print(f"Time to do the temporal analysis {(time.time() - start_time):.2f} s") 
