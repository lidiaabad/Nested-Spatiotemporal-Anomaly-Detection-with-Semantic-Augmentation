
import os, datetime, torch, copy
import numpy as np
import pandas as pd
from pathlib import Path
from data.pilots_preprocessing import TimeSeriesData, TSDataset
from utils.helpers_setup import build_experiment_path
from torch.utils.data import DataLoader


    
def build_loaders(base_dataset, args, positive=None):
    """
    Build train val-test loaders and return also labels for which EK rules = 1, feature names, labeles for injected anoamlies and the base dataset
    """
    loaders = {}
    pre_ds = TSDataset(mode=args.mode_pt, data=base_dataset.raw_df_with_alive.copy(),
                            feature_names=base_dataset.raw_df_with_alive.columns.tolist(), 
                            lb=args.lb,val_ratio=args.val_ratio, test_ratio=args.test_ratio, 
                            split_order=args.split_order,positive=positive,augment=args.augment,
                            figure_path=os.path.join(build_experiment_path(args), "split_plot"), 
                            inject_on_init=True, anom_freq=args.anom_freq, anom_type=args.anom_type, anom_sev=args.anom_sev)

    pre_ds.mode = "train"
    pre_ds.current_data = pre_ds.all_data["train"]
    loaders["train"] = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=False) 
    pre_ds.mode = "val"
    pre_ds.current_data = pre_ds.all_data["val"]
    loaders["val"] = DataLoader(pre_ds, batch_size=args.batch_size,shuffle=False)
    pre_ds.mode = "test"
    pre_ds.current_data = pre_ds.all_data["test"]
    loaders["test"] = DataLoader(pre_ds, batch_size=1, shuffle=False)
    
    test_ek = pre_ds.test_ek_mask
    feature_names = pre_ds.feature_names
    inj_anom_index=pre_ds.injected_anomalies_labels

    return loaders, test_ek, feature_names, inj_anom_index, pre_ds


def load_pilot_entity(args):
    " Get the base dataset using the threshold to define the ek labels and build the experiment path"
    
    site_name = Path(args.data_path).name.lower()
    SITE_DATA =  {
        "delos": { 
            "R1a_rain_1h": 10, "R1b_rain_3h": 20,  "R2_wind": 12,"R3_hum_th": 75, "R3_cross": 9,  
        },
        "baltanas": { 
            "R1_ppv": 0.25, "R2_hum_daily_range": 10, "R3_gnss_change": 0.1, "R4a_soil_high": 70, "R4b_soil_daily_range": 3, "R5a_airflow_const": 0.000001, "R5b_airflow_range": 0.05, 
        },
        
        "lucretili": {
            "R1a_crack_step": 0.6, "R1b_crack_offset": 0.5,"R2a_tilt_step": 0.2, "R2b_tilt_offset": 0.2,  "R3_vib_zscore": 7,"R4a_hum_spike_6h": 97, "R4b_hum_persist_7d": 90, "R4c_hum_rapid_1h": 20, 
            "R5_visitors_peak": 20, "R6a_rain_1h": 5, "R6a_rain_24h": 20, "R6b_rain_cum7d": 50, "R7a_wind_peak_ratio": 15, "R8_baro_range": 600, 
            "R9a_temp_high_st": 35, "R9b_temp_high_lt": 25, "R9c_temp_drop_st": 30, "R9d_temp_drop_lt": 5, "R10_solar_zscore": 3, 
        },
        "ranverso": { "R1_hum_inside": 80,  'R2_rain_7d_diff': 20,  "R3_temp_nodiff": 5,  "R4a_soil_high": 60, "R4b_soil_low": 5,
        },
        "schenkenberg": { "R1_temp_low": 5, "R1_hum_high": 90,  "R2_crack_step": 0.1, "R3_vib_zscore": 3,  "R4_pga": 3,  
            #"R5a_soil_high": 95,  "R5b_soil_range": 1,
        },
    }
    ths = dict(SITE_DATA.get(site_name.lower(), {}))
    freq="30min"  
    
    dataset = TimeSeriesData(args.data_path, freq, args.ek, threshold_overrides= ths) 
    
    return {
        "entity": None,
        "dataset": dataset,
        "experiment_path": build_experiment_path(args),
        "time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }


def load_data(args, public_datasets):
   return load_pilot_entity(args)