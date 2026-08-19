import pandas as pd
import numpy as np
import torch, os, random, csv, time, re
from functools import reduce
from utils import helpers_plots as hp
from data.ekrules import RuleEngine
from torch.utils.data import Dataset
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import STL
import shutil
import pickle
import copy
from collections import Counter, deque

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import spearmanr

'''
Main data processing file for the 5 pilot cases of ARGUS.
Contains TimeSeriesData (base dataset) and TSDataset (model-ready dataset).
Downstream / PU classes removed: only the reconstruction+forecast pretext task is used.
'''


class TimeSeriesData:
    '''
    Base Time Series Dataset.
    - Reads from folder_path containing all sensor/variable CSVs for a pilot.
    - Merges all data respecting variable and sensor name.
    - Builds alive_df indicating which features are live (mask for loss).
    - Resamples data to freq.
    - Defines anomalous instances via expert rules.
    '''

    def __init__(self, folder_path, freq, ek, threshold_overrides=None):

        self.folder_path = folder_path
        self.ek = ek
        self.freq = freq
        self.threshold_overrides = threshold_overrides or {}

        self.split_ranges = {}
        print(f"Using data from {self.folder_path}")

        # Load raw merged data
        raw_merged, _, labels = self.load_raw_data()
        # Contextual corrections (before frozen detection, no cyclic yet)
        tr_merged = self.apply_contextual_adjustments(raw_merged)

        if "baltanas" in self.folder_path.lower():
            '''
            # Keep only the functioning sensors after the restoration works
            keep_ids = {"159313", "400212", "149443"}

            cols_to_drop = []
            for col in tr_merged.columns:
                sid = self.extract_id(col)
                if sid and sid not in keep_ids:
                    cols_to_drop.append(col)
            tr_merged.drop(columns=cols_to_drop, inplace=True)
            print("WARNING: Model only using functioning sensors after restoration work (started on 15/04/2026). Alternative: use only data until then." )
            '''
            # End-of-monitoring cutoff for the Baltanas pilot (restauration works)
            cutoff = pd.Timestamp("2026-04-14 23:59:59")
            tr_merged = tr_merged.loc[tr_merged.index <= cutoff]
            print(f"Baltanas truncated at {cutoff}")

        # Detect frozen periods on raw data (no cyclic columns present yet).
        # NOTE: this no longer drops low-coverage columns -- it only masks
        # frozen periods to NaN and returns the alive mask. Dropping happens
        # AFTER resampling (see drop_low_alive_features below), because raw
        # merged timestamps come from a union of every sensor's native
        # cadence and unfairly penalize sparsely-sampled sensors.
        cleaned_raw, alive_df = self.build_frozen_mask(tr_merged)

        # Resample AFTER frozen detection
        resampled, resampled_alive = self.resample(cleaned_raw, alive_df)
        resampled_alive = resampled_alive.reindex(resampled.index).fillna(0).astype(int)

        # Drop low-coverage features on the RESAMPLED grid, respecting each
        # column's own registered valid window (e.g. _pre/_post regimes,
        # or a manually NaN'd sub-range like Ranverso 168442 humidity).
        resampled, resampled_alive = self.drop_low_alive_features(
            resampled, resampled_alive, min_alive_pct=10.0
        )

        # Unit conversion + dedup
        resampled = (self.convert_sensor_units(resampled).sort_index().groupby(level=0).last())

        # cyclic
        resampled_no_cyc = resampled.copy()
        resampled = self.add_cyclic(resampled)
        self.raw_df = resampled
        self.raw_df_with_alive = pd.concat([resampled, resampled_alive], axis=1)

        self.feature_base_names = resampled_no_cyc.columns.tolist()
        self.feature_names = resampled.columns.tolist()
        self.sup_labels = labels
        self.events_full = None

        self.engine = RuleEngine(self.raw_df, self.threshold_overrides)

        if self.ek:
            self.ek_rules = self.expert_knowledge_labels()
            row_max = self.ek_rules[0].max(axis=1)
        else:
            self.ek_rules = None

    def add_cyclic(self, df):
        """Add sin/cos cyclic encodings of hour, day-of-week and day-of-year.
        Called AFTER resampling, once the index is a clean DatetimeIndex."""
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            dow = df.index.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            dayofyear = df.index.dayofyear
            df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365)
            df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365)
        return df

    def convert_sensor_units(self, df):
        """Convert each column from its raw sensor reading (voltage, raw ADC
        counts, etc.) to physical units, based on keyword matches in the
        column name. Conversion formulas are sensor-datasheet specific.
        All transformations are based on sensor specifications
        """
        out = df.copy()
        converted = []

        for col in df.columns:

            name = col.lower()

            ssid_match = re.match(r"([A-Za-z0-9]{6})", col)
            ssid = ssid_match.group(1) if ssid_match else None

            variable = re.sub(r"\d+", "", name).replace("_", "").strip()

            def mark():
                if ssid is not None:
                    converted.append(f"{ssid} | {variable}")

            raw = pd.to_numeric(df[col], errors="coerce")

            if "airflow" in name or "af" in name:
                #V = raw # * 3.3 / 2048
                if ssid in ["277904","467704"]: 
                    raw=raw*3.3/2048
                out[col] = (-2.297 * raw**4 + 11.854 * raw**3 - 22.073 * raw**2 + 19.357 * raw - 5.498)
                mark()
                continue
            

            if "soilmoisture" in name or "moisture" in name:
                if ssid in ["149423", "149368"]:
                    out[col] = (1 - raw / 1000) * 100
                else:
                    out[col] = (1 - raw / 3.3) * 100
                mark()
                continue

            if "lux" in name:
                out[col] = raw * 1200
                mark()
                continue

            if "press" in name:
                if raw.max() <= 5:
                    out[col] = 65 + 8 * raw
                mark()
                continue

            if "hum" in name:
                if raw.max() <= 5:
                    out[col] = raw * 100 / 5
                mark()
                continue

            if "therm" in name:
                out[col] = 1 / (1.1293e-3 + 2.341e-4 * np.log(raw) + 8.7755e-8 * (np.log(raw))**3) - 273.2
                mark()
                continue

            if "pyr" in name:
                out[col] = raw * 1500 / 5
                mark()
                continue

            if "crack" in name:
                out[col] = raw * 250 / 5
                mark()
                continue

            if "rain_out" in name:
                out[pd.col] = raw.diff().clip(lower=0) * 0.2
                out[col] = out[col].fillna(0)
                mark()
                continue

            if "so2" in name:
                out[col] = np.where(raw > 0.6, (10 / 0.9) * (raw - 0.6), 0)
                mark()
                continue

            if "co" in name and "counter" not in name and "cos" not in name and "comp" not in name and "conv" not in name:
                out[col] = np.where(raw > 0, 10 ** ((np.log10((3.3 / raw - 1) / 2) - 1.7) / -0.77), 0)
                mark()
                continue

            if "no2" in name:
                out[col] = np.where(raw > 0, 10 ** ((np.log10((3.3 / raw - 1) / 2) + 0.1) / 0.8), 0)
                mark()
                continue

        out = out.sort_index()
        return out

    def load_raw_data(self):
        """Walk folder_path, read every CSV, group columns by sensor ID (first
        token of the filename) and merge everything into a single wide
        DataFrame indexed by datetime. Also splits off any '*label*' columns
        into a separate supervised-label array."""
        sensor_groups = {}

        for root, _, files in os.walk(self.folder_path):
            for fname in files:
                if not fname.lower().endswith(".csv"):
                    continue
                path = os.path.join(root, fname)
                try:
                    base = fname.lower().split(".")[0]
                    sensor_id = re.split(r'[_-]', base, maxsplit=1)[0]
                    if self.is_block_style_csv(path):
                        raw_dfs = [self.read_block_csv(path)]
                    else:
                        raw_dfs = [self.read_csv_flexible(path)]
                    for raw_df in raw_dfs:
                        df = self.clean_single_df(raw_df, fname)
                        if df is not None and not df.empty:
                            df.columns = (
                                df.columns
                                .str.replace("\ufeff", "", regex=False)
                                .str.replace('"', "", regex=False)
                                .str.replace("'", "", regex=False)
                                .str.replace("\u200b", "", regex=False)
                                .str.replace("\u200c", "", regex=False)
                                .str.replace("\u200d", "", regex=False)
                                .str.strip()
                            )
                            sensor_groups.setdefault(sensor_id, [])
                            sensor_groups[sensor_id].append(df)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
                    continue

        dfs = []
        for sid, df_list in sensor_groups.items():
            col_groups = {}
            for df in df_list:
                for col in df.columns:
                    col_groups.setdefault(col, [])
                    col_groups[col].append(df[[col]])
            merged_columns = []
            for col_name, col_dfs in col_groups.items():
                col_df = pd.concat(col_dfs, axis=0)
                col_df = col_df[~col_df.index.duplicated(keep="first")]
                col_df = col_df.sort_index()
                merged_columns.append(col_df)
            sensor_df = pd.concat(merged_columns, axis=1)
            sensor_df = sensor_df.sort_index()
            dfs.append(sensor_df)

        merged = pd.concat(dfs, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.sort_index(inplace=True)

        label_cols = [c for c in merged.columns if "label" in c.lower()]
        labels = None
        if label_cols:
            labels = merged[label_cols].any(axis=1).astype(int).values
            merged = merged.drop(columns=label_cols)

        feature_names = merged.columns.tolist()
        return merged, feature_names, labels

    def read_csv_flexible(self, path):
        """Best-effort CSV reader: tries pandas' automatic separator sniffing
        first, then falls back to a manual sep/decimal grid search for
        malformed exports (mixed ';'/',' separators and decimal commas)."""
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = None
        if df is None or df.shape[1] == 1:
            for sep in [";", ",", "\t", "|"]:
                for dec in [".", ","]:
                    try:
                        df = pd.read_csv(path, sep=sep, decimal=dec)
                        if df.shape[1] > 1:
                            break
                    except Exception:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
        if df is None or df.shape[1] == 1:
            raise ValueError(f"Could not read file: {path}")
        first_col = df.columns[0]
        if df[first_col].astype(str).str.contains(";").any():
            split = df[first_col].astype(str).str.split(";", expand=True)
            df[first_col] = split[0]
            if split.shape[1] > 1:
                df[df.columns[1]] = split[1]
        for col in df.columns[1:]:
            if df[col].dtype == object:
                df[col] = (df[col].astype(str).str.replace(",", ".", regex=False))
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def is_block_style_csv(self, path):
        """Detect the alternate 'block-style' CSV export (variable name on
        its own header line, followed by a Datetime,value block) used by
        some loggers, identified by Spanish column keywords."""
        keywords = ["Temperatura", "Humedad"]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):
                line = f.readline().strip()
                if any(k in line for k in keywords):
                    return True
        return False

    def read_block_csv(self, path):
        """Parse the block-style CSV format: repeated
        [variable_name header] -> [Datetime, value rows] blocks, merged into
        a single wide DataFrame indexed by datetime."""
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line)

        blocks = []
        current_var = None
        current_data = []
        date_pattern = r"\d{2}/\d{2}/\d{4}"

        for row in rows:
            if "," not in row and not re.search(date_pattern, row):
                if current_var and current_data:
                    blocks.append((current_var, current_data))
                current_var = (
                    row.replace("\ufeff", "").replace("\u200b", "")
                    .replace("\u200c", "").replace("\u200d", "").strip())
                current_data = []
                continue
            if "Tiempo" in row:
                continue
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 2 and re.search(date_pattern, parts[0]):
                current_data.append(parts[:2])

        if current_var and current_data:
            blocks.append((current_var, current_data))

        clean_blocks = []
        for var, data in blocks:
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                clean_blocks.append((var, data))
            else:
                print(f"Skipping invalid block: {var} -> {data}")
        blocks = clean_blocks

        dfs = []
        for var, data in blocks:
            df = pd.DataFrame(data, columns=["Datetime", var])
            df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True, errors="coerce")
            df[var] = pd.to_numeric(df[var].str.replace(",", "."), errors="coerce")
            df = df.dropna(subset=["Datetime", var])
            df = df.set_index("Datetime")
            dfs.append(df)

        if len(dfs) == 1:
            return dfs[0]

        full_index = dfs[0].index
        for df in dfs[1:]:
            full_index = full_index.union(df.index)
        aligned = [df.reindex(full_index) for df in dfs]
        merged_df = pd.concat(aligned, axis=1)
        return merged_df

    def clean_single_df(self, df, fname):
        """Normalize a single raw file's DataFrame: ensure a proper
        DatetimeIndex, namespace its columns by sensor prefix, dedupe by
        timestamp. Returns None if nothing usable is left."""
        if df is None or df.empty:
            return None
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df.index.name = "datetime"
        else:
            df = self.normalize_datetime_column(df, fname)
            if df is None:
                return None
            df["datetime"] = self.parse_datetime_series(df["datetime"])
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        df = self.namespace_and_clean(df, fname)
        if df is None or df.empty:
            return None
        df = df.groupby(df.index).first().sort_index()
        return df

    def normalize_datetime_column(self, df, fname):
        """Find and rename whichever column looks like a datetime column
        (matches 'date'/'time'/'fecha' in its name) to 'datetime'."""
        df.columns = (
            df.columns
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace("\u200b", "", regex=False)
            .str.replace("\u200c", "", regex=False)
            .str.replace("\u200d", "", regex=False)
            .str.strip()
            .str.lower()
        )
        dt_candidates = [c for c in df.columns if "date" in c or "time" in c or "fecha" in c]
        if not dt_candidates:
            print(f"[ERROR] No datetime column found in {fname}: {df.columns}")
            return None
        dt_col = dt_candidates[0]
        df = df.rename(columns={dt_col: "datetime"})
        return df

    def parse_datetime_series(self, s):
        """Try a fixed list of known datetime formats before falling back to
        pandas' dayfirst-aware generic parser."""
        s = (s.astype(str).str.replace("\u00a0", " ", regex=False).str.strip())
        if s.str.contains(";").any():
            s = s.str.split(";", n=1).str[0]
        formats = [
            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S"
        ]
        for fmt in formats:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().any():
                return parsed
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

    def namespace_and_clean(self, df, fname):
        """Coerce all columns to numeric, drop all-NaN columns, and prefix
        column names with the sensor/site identifier parsed from the
        filename (skipped for the 5 known pilot sites, whose CSVs already
        ship with self-describing column names)."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.apply(lambda col: pd.to_numeric(
            col.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"))
        df = df.dropna(axis=1, how="all")
        if df.empty:
            return pd.DataFrame()
        df = df.groupby(df.index).first()
        df = df.sort_index()
        base = os.path.splitext(fname)[0]
        base_lower = base.lower()
        prefix = base_lower.split("_")[0]
        KNOWN_SITES = ["baltanas", "delos", "lucretili", "ranverso", "schenkenberg"]
        if any(prefix.startswith(site) for site in KNOWN_SITES):
            df.columns = [c.lower() for c in df.columns]
        else:
            df.columns = [f"{prefix}_{c.lower()}" for c in df.columns]
        return df

    def extract_id(self, col):
        """
        Extract base sensor ID (first 6 alphanumeric chars) and distinguish
        _pre/_post splits so they are treated as different sensors.
        Returns None if the column doesn't start with a well-formed sensor ID.
        """
        parts = col.split("_")
        base = parts[0]

        # base ID must be 6 alphanumeric chars
        if not re.fullmatch(r"[A-Za-z0-9]{6}", base):
            return None

        # suffix handling --> so pre and post are treated as independent sensors
        if col.endswith("_pre"):
            return base + "_pre"
        if col.endswith("_post"):
            return base + "_post"

        return base

    def register_valid_window(self, col, start=None, end=None):
        """
        Declares that `col` should only be judged for coverage/alive%
        within [start, end) (half-open; either bound may be None meaning
        unbounded), instead of over the full dataset. Call this any time
        you intentionally NaN-out or split part of a column for a
        structural reason (regime change, sensor swap, known bad period)
        rather than because the sensor is actually dead there.

        Stored as timestamps (not a boolean Series tied to one index) so
        it can be evaluated against BOTH the raw and the resampled index.
        Multiple calls for the same col intersect their windows.
        """
        new_start = pd.Timestamp(start) if start is not None else None
        new_end = pd.Timestamp(end) if end is not None else None
        if col in self.split_ranges:
            old_start, old_end = self.split_ranges[col]
            if new_start is not None:
                old_start = new_start if old_start is None else max(old_start, new_start)
            if new_end is not None:
                old_end = new_end if old_end is None else min(old_end, new_end)
            self.split_ranges[col] = (old_start, old_end)
        else:
            self.split_ranges[col] = (new_start, new_end)

    def _valid_mask_for(self, col, index):
        """Boolean mask over `index` marking where col's registered valid
        window applies (True everywhere if no window was registered)."""
        if col not in self.split_ranges:
            return pd.Series(True, index=index)
        start, end = self.split_ranges[col]
        mask = pd.Series(True, index=index)
        if start is not None:
            mask &= (index >= start)
        if end is not None:
            mask &= (index < end)
        return mask

    def apply_contextual_adjustments(self, df):
        """
        Apply manual, site-specific corrections that must happen BEFORE
        frozen-period detection: thermistor invalid-range removal, negative
        light/UV clipping, wind-direction cyclic encoding, and each site's
        own list of known sensor issues (splits, drops, unit fixes).
        Every per-site block below encodes a specific, manually diagnosed
        data-quality issue
        """
        df = df.copy()
        folder = self.folder_path.lower()

        def split_sensor(sensor_id, cutoff, elim_date=None):
            """Split one sensor's column into _pre/_post halves around
            `cutoff` (e.g. a hardware swap or relocation), optionally
            blanking a transition period up to/from `elim_date`."""
            for col in list(df.columns):
                sid_full = self.extract_id(col)
                if sid_full is None:
                    continue
                sid_base = sid_full[:6]
                if sid_base != sensor_id:
                    continue

                pre_col = col + "_pre"
                post_col = col + "_post"

                pre_regime = df.index < cutoff
                post_regime = df.index >= cutoff

                df[pre_col] = df[col].where(pre_regime)
                df[post_col] = df[col].where(post_regime)

                pre_end = cutoff
                post_start = cutoff

                if elim_date is not None:
                    df.loc[df.index >= elim_date, pre_col] = np.nan
                    df.loc[df.index < elim_date, post_col] = np.nan
                    pre_end = min(cutoff, elim_date)
                    post_start = max(cutoff, elim_date)

                # remember each half's own valid window so alive% isn't
                # diluted by the NaNs that belong to the other half
                self.register_valid_window(pre_col, end=pre_end)
                self.register_valid_window(post_col, start=post_start)

                df.drop(columns=[col], inplace=True)

        # Thermistor invalid range fix
        for col in df.columns:
            if "therm" in col.lower():
                v = pd.to_numeric(df[col], errors="coerce")
                invalid_thr = v > 100000 # some threshmistor broke and went over this value
                invalid_final = pd.Series(False, index=df.index)
                in_bad_block = False
                block_start = None
                invalid_intervals = []
                for i in range(len(df)):
                    ts = df.index[i]
                    if not in_bad_block:
                        if invalid_thr.iat[i]:
                            in_bad_block = True
                            block_start = ts
                            invalid_final.iat[i] = True
                    else:
                        invalid_final.iat[i] = True
                        if not invalid_thr.iat[i]:
                            ts_end = ts + pd.Timedelta(days=1)
                            window = (~invalid_thr).loc[ts:ts_end]
                            if window.all():
                                in_bad_block = False
                                block_end = ts
                                invalid_intervals.append((block_start, block_end))
                                block_start = None
                if in_bad_block and block_start is not None:
                    block_end = df.index[-1]
                    invalid_intervals.append((block_start, block_end))
                if invalid_final.any():
                    df.loc[invalid_final, col] = np.nan

        # Negative light and UV values are sensor artefacts
        for col in df.columns:
            if ("light" in col.lower()) or ("uv" in col.lower()):
                neg_mask = df[col] < 0
                if neg_mask.any():
                    df.loc[neg_mask, col] = np.nan

        # Drop rain accumulation columns (redundant with rain rate)
        acum_cols = [c for c in df.columns if "rain_acc" in c.lower()]
        if acum_cols:
            df.drop(columns=acum_cols, inplace=True)

        # Wind direction to cyclic (angle -> sin/cos) since raw degrees are
        # not meaningful to a model as a linear feature
        dir_cols = [c for c in df.columns if "dir" in c.lower()]
        for col in dir_cols:
            radians = np.deg2rad(df[col])
            df[col + "_sin"] = np.sin(radians)
            df[col + "_cos"] = np.cos(radians)
            df.drop(columns=[col], inplace=True)

        if "delos" in folder:
            # Sensor 149678 changed monitored surface (plaster wall -> mosaic) on this date, so pre/post readings are not comparable
            #elim_date = pd.Timestamp("2026-02-11")
            #for sid in ["149678"]:
            #    split_sensor(sensor_id=sid, cutoff=elim_date, elim_date=elim_date)

            for col in df.columns:
                sid = self.extract_id(col)
                # Sensor 149423 dropped entirely -- corrupted data for the whole monitoring period (excluded from paper results)
                if sid in ["149423"]:
                    df.drop(columns=col, inplace=True)
                    #print(f"Sensor {sid} removed due to corrupted data in {col}")

        if "baltanas" in folder:
            shelly_ids = ["caa50b", "7020c8", "caa416", "701ee3"]  #Shelly-brand humidity/temp sensors, handled separately below
            diff_ids = ["45720d", "4d7f09"]  # Sensors reporting a monotonic counter, differenced to a rate
            for col in df.columns:
                sid = self.extract_id(col)
                if sid in ["130314", "130432"] and "water_potential" in col.lower():
                    # Negative water potential readings are sensor error; ffill bridges the gap
                    mask_neg = df[col] < 0
                    if mask_neg.any():
                        df.loc[mask_neg, col] = np.nan
                        df[col] = df[col].ffill()
            for col in df.columns:
                sid = self.extract_id(col)
                if sid in shelly_ids:
                    if "hum" in col.lower() or "humedad" in col.lower():
                        # Humidity==0 readings are dropouts; blank a +/-3-sample window around each one for the whole sensor group
                        zero_idx = df.index[df[col] == 0]
                        for idx in zero_idx:
                            pos = df.index.get_loc(idx)
                            start = max(pos - 3, 0)
                            end = min(pos + 3, len(df) - 1)
                            drop_idx = df.index[start:end + 1]
                            sensor_cols = [c for c in df.columns if self.extract_id(c) == sid]
                            df.loc[drop_idx, sensor_cols] = np.nan
                    if sid != "701ee3":  # 701ee3 does not exhibit these two failure modes bc outside
                        # Temperature stuck at 0 is a sensor fault
                        if "temp" in col.lower():
                            mask = df[col] == 0
                            if mask.any():
                                df.loc[mask, col] = np.nan
                                #print(f"Temperature=0 removed for {col}")
                        # Humidity < 70 treated as sensor error (stuck low value,observed around 50/66) for this sensor group specifically
                        if "hum" in col.lower() or "humedad" in col.lower():
                            mask = df[col] < 70
                            if mask.any():
                                df.loc[mask, col] = np.nan
            df.interpolate(method="time", inplace=True)
            cols_to_drop = [c for c in df.columns if "duration" in c.lower() or "alarm" in c.lower()]
            df.drop(columns=cols_to_drop, inplace=True)
            for col in df.columns:
                sid = self.extract_id(col)
                if sid in diff_ids:
                    # Convert cumulative counter to incremental rate
                    df[col] = df[col].round().astype("Int64")
                    df[col] = df[col].diff().clip(lower=0)
                    df[col] = df[col].fillna(0)

        if "lucretili" in folder:
            # No automatic contextual correction currently applied for this site
            pass

        if "ranverso" in folder:
            # Sensor 149361 saturates during this window; rescaled to [0, 1] locally rather than dropped, since the relative shape is still usable
            offset_start = pd.Timestamp("2025-12-06 12:00")
            offset_end = pd.Timestamp("2026-02-02 00:00")
            elim_date2 = pd.Timestamp("2026-03-31")
            for col in df.columns:
                if self.extract_id(col) == "149361":
                    mask_time = (df.index >= offset_start) & (df.index <= offset_end)
                    if mask_time.any():
                        values = df.loc[mask_time, col]
                        v_min, v_max = values.min(), values.max()
                        df.loc[mask_time, col] = (values - v_min) / (v_max - v_min)
            for col in df.columns:
                sid = self.extract_id(col)
                if sid == "168442":
                    if "hum" in col.lower() or "humedad" in col.lower():
                        # This humidity channel is only meaningful from elim_date2 onward (e.g. new probe installed); 
                        mask = df.index < elim_date2
                        if mask.any():
                            df.loc[mask, col] = np.nan
                            self.register_valid_window(col, start=elim_date2)
            for col in df.columns:
                if self.extract_id(col) == "177315":
                    #Single-day known bad reading
                    mask_time = (df.index == elim_date2)
                    if mask_time.any():
                        df.loc[mask_time, col] = np.nan

                if "so2" in col.lower():
                    # SO2 sensor dropped for this site (no reliable calibration, negative)
                    df.drop(columns=col, inplace=True)
                    #print(f"Dropped column: {col}")

        if "schenkenberg" in folder:
            # Sensors 600345
            # dropped entirely -- corrupted data for the whole monitoring period, to be reviewed in a few months
            corrupted_sens = ["600345", "152914", "152555", "175480"] 
            for col in df.columns:
                sid = self.extract_id(col)
                if sid in corrupted_sens:
                    df.drop(columns=col, inplace=True)
                    print("[WARNING] removed", sid)
                    continue

                # RMS-derived columns are redundant with the raw signal
                if "rms" in col.lower():
                    df.drop(columns=col, inplace=True)

            for col in df.columns:
                sid = self.extract_id(col)
                # Sensor 152899's first 1000 samples are a known installation/warm-up artefact
                if sid == "152899":
                    df.loc[df.index[:1000], col] = np.nan

        return df

    def build_frozen_mask(self, df, verbose=True):
        """
        Detect frozen values per sensor (window-based).
        Returns:
            df_cleaned  -> with NaNs where frozen
            alive_df    -> 1 = alive, 0 = frozen

        NOTE: this no longer drops low-coverage columns. That decision is
        made later, on the RESAMPLED grid, by drop_low_alive_features --
        judging coverage on the raw merged index unfairly penalizes
        sparsely-sampled sensors, since the raw index is a union of every
        sensor's native timestamps across the whole site.
        """
        df = df.copy()
        frozen_mask = self.detect_frozen_periods_contextual(df)
        frozen_mask = frozen_mask.reindex(df.index)
        df_cleaned = df.mask(frozen_mask)

        alive_df = (~frozen_mask).astype(float)
        alive_df.index = df.index
        alive_df.columns = [c + "_alive" for c in df.columns]

        return df_cleaned, alive_df

    def drop_low_alive_features(self, df, alive_df, min_alive_pct=10.0, verbose=True):
        """
        Drops low-coverage features. Evaluated on whatever grid `df`/`alive_df`
        currently are (intended to be called on the RESAMPLED data), so a
        column's own reporting cadence isn't penalized by other sensors'
        finer raw timestamps. Any column with a registered valid window
        (self.split_ranges) is judged only within that window.
        """
        original_order = df.columns.tolist()
        cols_to_drop = []

        for col in df.columns:
            alive_col = col + "_alive"
            if alive_col not in alive_df.columns:
                continue
            alive_series = alive_df[alive_col]
            valid_mask = self._valid_mask_for(col, alive_series.index)

            if valid_mask.any():
                pct_alive = alive_series[valid_mask].mean() * 100
            else:
                pct_alive = 0.0

            if pct_alive < min_alive_pct:
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            alive_df = alive_df.drop(columns=[c + "_alive" for c in cols_to_drop])

        surviving_order = [c for c in original_order if c not in cols_to_drop]
        alive_order = [c + "_alive" for c in surviving_order]
        df = df.reindex(columns=surviving_order)
        alive_df = alive_df.reindex(columns=alive_order)

        return df, alive_df

    def detect_frozen_periods_contextual(self, df):
        """
        Detect frozen periods for each sensor group.
        Groups defined by first 6 alphanumeric characters (ID).
        Frozen when:
          1) ALL variables in group are NaN for >= max_gap
          2) Single-variable NaN gaps >= max_gap_single (all non-cyclic columns, not just thermistors)
          3) Manual rules per site
          4) Raw NaN, for any column with a registered valid window
             (self.split_ranges) -- this includes _pre/_post splits AND
             any manually NaN'd sub-range registered via
             register_valid_window(). Without this, a column that's
             intentionally blanked outside its valid window (e.g. Ranverso
             168442 humidity before elim_date2) would show alive=1 there,
             since nothing else marks it frozen.
        """
        frozen = pd.DataFrame(False, index=df.index, columns=df.columns)
        groups = {}
        for col in df.columns:
            sid = self.extract_id(col)
            if sid is None:
                continue
            groups.setdefault(sid, []).append(col)

        folder = self.folder_path.lower()

        def mark_frozen(sensor_ids, start=None, end=None):
            """Mark all columns belonging to any of `sensor_ids` as frozen
            over [start, end] (either bound optional -> open-ended)."""
            for col in df.columns:
                sid = self.extract_id(col)
                if sid is None:
                    continue
                if any(sid == s or col.startswith(s) for s in sensor_ids):
                    if start and end:
                        frozen.loc[start:end, col] = True
                    elif end:
                        frozen.loc[:end, col] = True
                    elif start:
                        frozen.loc[start:, col] = True

        # Sensors malfunction 
        if "delos" in folder:
            start = pd.Timestamp("2025-10-26")  
            end = pd.Timestamp("2026-02-28")
            mark_frozen(['149529'], start=start, end=end)
            
            start = pd.Timestamp("2025-10-22")  
            end = pd.Timestamp("2025-10-29")
            mark_frozen(['300548'], start=start, end=end)
            
            end1 = pd.Timestamp("2025-11-29")  # Several problems with this sensor
            mark_frozen(['50c42f'], end=end1)
            start1 = pd.Timestamp("2025-11-16")  
            end2 = pd.Timestamp("2025-12-13")
            mark_frozen(['50c42f'], start=start1, end=end2)
            start2 = pd.Timestamp("2025-12-26") 
            mark_frozen(['50c42f'], start=start2)
            
            cutoff = pd.Timestamp("2026-03-23")  # Bad connection: fixed on this date
            mark_frozen(['168439'], end=cutoff)
            
            cutoff = pd.Timestamp("2026-02-10")  # New addition
            mark_frozen(['3001F4'], end=cutoff)
            
            start = pd.Timestamp("2026-02-02")  # No connection for a week 
            end = pd.Timestamp("2026-02-11")
            mark_frozen(['151815'], start=start, end=end)
            
            end = pd.Timestamp("2025-10-21") 
            mark_frozen(['151805'], end=end)
            
            end = pd.Timestamp("2026-02-12")
            mark_frozen(['149678'], end=end)
    
    
        if "baltanas" in folder:
            baltanas_sensors = [
                '45720d', '4d7f09', '277904', '467704',
                '149367', '149780', '149402', '149421',
                '177322', '149265', '130314', '130432', '149443',
                '151810']
            cutoff = pd.Timestamp("2025-11-29")  # Predeployment
            mark_frozen(baltanas_sensors, end=cutoff)
            
            pir_airflows = ['277904', '467704']
            start = pd.Timestamp("2025-12-18")  # PIR airflow sensors failed from this date
            mark_frozen(pir_airflows, start=start)
            
            end = pd.Timestamp("2026-01-05")  
            mark_frozen(['45720d'], start=start, end=end) # counter failed before because of connection
            
            start = pd.Timestamp("2026-01-16")  
            end = pd.Timestamp("2026-04-16")
            mark_frozen(['151810'], start=start, end=end) # problems with batteries 
            
            baltanas_sensors2 = [
                '7020c8', 'caa50b', '130432', '130314',
                '149265', '177322', '159347', '149421',
                '149402', '149780', '149367']
            start = pd.Timestamp("2026-04-15")  # sensors eliminated on the first restoration date
            mark_frozen(baltanas_sensors2, start=start)
            exterior_lagar = ['701ee3', 'caa416']
            start = pd.Timestamp("2026-04-15")  # sensors eliminated on the first restoration date
            end = pd.Timestamp("2026-04-15")
            mark_frozen(exterior_lagar, start=start, end=end)
            
            start = pd.Timestamp("2026-05-12") 
            mark_frozen(['159164'], start=start) # restoration broke sensor 

        if "lucretili" in folder:
            start = pd.Timestamp("2025-12-18")  
            end = pd.Timestamp("2026-02-06")
            mark_frozen(['152608', '177281', '152816', '177253', '177266'], start=start, end=end) # probelms with connectiond 
            
            start = pd.Timestamp("2025-12-18")  
            end = pd.Timestamp("2026-02-07")
            mark_frozen(['177239'], start=start, end=end) # probelms with connectiond 
            
            end = pd.Timestamp("2025-12-01")  
            mark_frozen(['177281'], end=end)
            start = pd.Timestamp("2026-03-07")  
            end = pd.Timestamp("2026-03-12")
            mark_frozen(['177281'], start=start, end=end) # disconnected 
            
            start = pd.Timestamp("2025-12-12")  
            end = pd.Timestamp("2026-02-06")
            mark_frozen(['153061'], start=start, end=end) # disconnected/batteries
            
            cutoff = pd.Timestamp("2025-12-06")  
            mark_frozen(['152608'], end=cutoff) # added
            
            start = pd.Timestamp("2025-12-18")  
            end = pd.Timestamp("2026-04-22")
            mark_frozen(['177320'], start=start, end=end)
            
            start = pd.Timestamp("2026-02-05")  # Comunication failure
            end = pd.Timestamp("2026-03-03")
            mark_frozen(['174156', '450308', '300286'], start=start, end=end)
            
            start = pd.Timestamp("2026-02-12")  # Comunication failure
            end = pd.Timestamp("2026-03-03")
            mark_frozen(['177287', '177237'], start=start, end=end)
            
            cutoff = pd.Timestamp("2025-10-31")  
            mark_frozen(['450308', '300286'], end=cutoff) #added
            
            end1 = pd.Timestamp("2025-11-23")  # Vandalism: broken sensors
            mark_frozen(['177310'], end=end1)
            start1 = pd.Timestamp("2025-12-21") 
            end2 = pd.Timestamp("2026-02-04")
            mark_frozen(['177310'], start=start1, end=end2)
            start2 = pd.Timestamp("2026-03-02")  
            end3 = pd.Timestamp("2026-03-07")
            mark_frozen(['177310'], start=start2, end=end3)
            cutoff = pd.Timestamp("2026-01-28")  
            mark_frozen(['177325'], end=cutoff)

        if "ranverso" in folder:
            cutoff = pd.Timestamp("2025-11-13")  # added
            mark_frozen(['143276', '3d6617', '170894', '177315'], end=cutoff)
            
            start = pd.Timestamp("2026-02-02")  # lost communication
            end = pd.Timestamp("2026-02-04")
            mark_frozen(['170894'], start=start, end=end)
            
            cutoff = pd.Timestamp("2025-12-06")  # added
            mark_frozen(['149361'], end=cutoff)
            
            start = pd.Timestamp("2026-02-02")  # lost communication
            end = pd.Timestamp("2026-03-06")
            mark_frozen(['149361'], start=start, end=end)
            
            start = pd.Timestamp("2026-02-02")  # added
            mark_frozen(['3d6617'], start=start)
            
            cutoff = pd.Timestamp("2025-12-11")  # broke
            mark_frozen(['26680F'], end=cutoff)
            
            end1 = pd.Timestamp("2025-11-20")  # added
            mark_frozen(['359111'], end=end1)
            
            start1 = pd.Timestamp("2025-12-18")  # corrupted
            end2 = pd.Timestamp("2026-01-19")
            mark_frozen(['359111'], start=start1, end=end2)
        
        if "schenkenberg" in folder:
            end = pd.Timestamp("2026-06-10")  # trials before 10/06 are non-reliable in accs
            mark_frozen(['159070', '152555', '152479', '175480'], end=end)

        # Any column with a registered valid window (split_sensor _pre/_post,
        # OR a manually NaN'd sub-range via register_valid_window) must have
        # its raw NaN folded into `frozen`. Otherwise a NaN outside the
        # window (or a NaN that no other rule catches) silently reads as
        # alive=1.
        for col in df.columns:
            if col.endswith("_pre") or col.endswith("_post") or col in self.split_ranges:
                frozen[col] = frozen[col] | df[col].isna()

        # Automatic freeze: any sensor group with ALL variables NaN for a
        # sustained period is considered frozen (offline), regardless of site.
        max_gap = pd.Timedelta(days=3)  # because some sensors data every 2 days

        for sid, cols in groups.items():
            sensor_data = df[cols]
            group_nan = sensor_data.isna().all(axis=1)
            nan_groups = (group_nan != group_nan.shift()).cumsum()
            for grp, mask in group_nan.groupby(nan_groups):
                if not mask.iloc[0]:
                    continue
                start = mask.index[0]
                end = mask.index[-1]
                if end - start >= max_gap:
                    extended_end = end + pd.Timedelta(days=1)
                    frozen.loc[start:extended_end, cols] = True

        for col in df.columns:
            if "therm" in col.lower():
                v = pd.to_numeric(df[col], errors="coerce")
                invalid_thr = v > 100000  # Same threshold as apply_contextual_adjustments

                invalid_final = pd.Series(False, index=df.index)
                in_bad_block = False
                block_start = None
                invalid_intervals = []

                for i in range(len(df)):
                    ts = df.index[i]

                    if not in_bad_block:
                        if invalid_thr.iat[i]:
                            in_bad_block = True
                            block_start = ts
                            invalid_final.iat[i] = True
                    else:
                        invalid_final.iat[i] = True
                        if not invalid_thr.iat[i]:
                            ts_end = ts + pd.Timedelta(days=1)
                            window = (~invalid_thr).loc[ts:ts_end]

                            if window.all():
                                in_bad_block = False
                                block_end = ts
                                invalid_intervals.append((block_start, block_end))
                                block_start = None

                if in_bad_block and block_start is not None:
                    block_end = df.index[-1]
                    invalid_intervals.append((block_start, block_end))

                if invalid_final.any():
                    df.loc[invalid_final, col] = np.nan

        return frozen

    def resample(self, df, alive_df):
        """
        Resample to self.freq. Counter-like columns (cumulative counts,
        identified by name pattern) are summed per bin; all other
        (continuous) columns are averaged and short gaps are time-interpolated.
        The alive mask is resampled by max (alive if any sub-sample was alive).
        """
        original_order = df.columns.tolist()
        alive_order = alive_df.columns.tolist()

        def is_counter(col):
            c = col.lower()
            return bool(re.search(r'(_in\b|_in_|^in_|counter)', c))

        counter_cols = [c for c in df.columns if is_counter(c)]
        cont_cols = [c for c in df.columns if c not in counter_cols]

        df_counter = df[counter_cols].resample(self.freq).sum(min_count=1)
        df_cont = df[cont_cols].resample(self.freq).mean()

        df_cont = df_cont.interpolate(method="time", limit=48) # 1 day limit 

        df_resampled = pd.concat([df_counter, df_cont], axis=1).sort_index()
        df_resampled = df_resampled.bfill(limit=2)  # 1 h limit

        alive_resampled = alive_df.resample(self.freq).max()
        alive_resampled = alive_resampled.reindex(alive_resampled.index)
        df_alive_resampled = alive_resampled.fillna(0).astype(int)

        df_resampled = df_resampled.reindex(columns=original_order)
        df_alive_resampled = df_alive_resampled.reindex(columns=alive_order)

        return df_resampled, df_alive_resampled

    def expert_knowledge_labels(self):
        """
        Run the site-specific RuleEngine to derive expert-knowledge (EK)
        anomaly labels, then restrict them to timestamps/columns that are
        actually 'alive' (a rule firing on frozen/dead data is not a real
        anomaly). Returns (scores_df, metrics_df, masks_df).
        """
        self.raw_df.index = pd.to_datetime(self.raw_df.index)
        self.raw_df = self.raw_df.sort_index()

        anomalies = pd.Series(False, index=self.raw_df.index, dtype=bool).sort_index()
        anomaly_sources = pd.DataFrame(False, index=self.raw_df.index, columns=self.raw_df.columns).sort_index()

        alive_cols = [c for c in self.raw_df_with_alive.columns if c.endswith("_alive")]
        alive_map = (
            self.raw_df_with_alive[alive_cols]
            .fillna(0)
            .rename(columns=lambda c: c.replace("_alive", ""))
            >= 0.5
        )

        folder_lower = self.folder_path.lower()

        if 'delos' in folder_lower:
            anomalies, rule_scores, metrics, rule_masks = \
                self.engine._apply_delos_rules(anomalies, anomaly_sources)
        elif 'baltanas' in folder_lower:
            anomalies, rule_scores, metrics, rule_masks = \
                self.engine._apply_baltanas_rules(anomalies, anomaly_sources)
        elif 'lucretili' in folder_lower:
            anomalies, rule_scores, metrics, rule_masks = \
                self.engine._apply_lucretili_rules(anomalies, anomaly_sources)
        elif 'ranverso' in folder_lower:
            anomalies, rule_scores, metrics, rule_masks = \
                self.engine._apply_ranverso_rules(anomalies, anomaly_sources)
        elif 'schenkenberg' in folder_lower:
            anomalies, rule_scores, metrics, rule_masks = \
                self.engine._apply_schenkenberg_rules(anomalies, anomaly_sources)
        else:
            raise ValueError(f"Could not detect site from path: {self.folder_path}")

        valid_anomalies = pd.Series(False, index=anomalies.index)
        for col in anomaly_sources.columns:
            if col in alive_map.columns:
                valid_anomalies |= anomaly_sources[col] & alive_map[col]
            else:
                valid_anomalies |= anomaly_sources[col]
        anomalies &= valid_anomalies

        scores_df = pd.DataFrame(rule_scores, index=self.raw_df.index)
        scores_df = scores_df.where(valid_anomalies, np.nan)

        metrics_df = pd.DataFrame(metrics, index=self.raw_df.index)
        metrics_df = metrics_df.where(valid_anomalies, np.nan)

        masks_df = pd.DataFrame(rule_masks, index=self.raw_df.index)
        masks_df = masks_df.where(valid_anomalies, False)

        return scores_df, metrics_df, masks_df

    def compute_rule_diagnostics(self, masks_df, metrics_df, noise_std=0.05):
        """
        For each EK rule, compute:
        - activation_rate: fraction of timestamps the rule fires on
        - burstiness: whether firings cluster together vs. spread evenly
        - noise_sensitivity: how often the rule's verdict flips under small
          injected Gaussian noise (n_trials re-runs), as a robustness check.
        Returns one row per rule.
        """
        diagnostics = {}

        def _run_rules(df):
            """Re-run the site's rule engine on a (possibly noised) copy of
            the data, returning only the boolean rule_masks."""
            anomalies = pd.Series(False, index=df.index)
            anomaly_sources = {c: pd.Series(False, index=df.index) for c in df.columns}
            folder_lower = self.folder_path.lower()
            if 'delos' in folder_lower:
                _, _, _, rule_masks = self.engine._apply_delos_rules(anomalies.copy(), anomaly_sources, df=df, verbose=False)
            elif 'baltanas' in folder_lower:
                _, _, _, rule_masks = self.engine._apply_baltanas_rules(anomalies.copy(), anomaly_sources, df=df, verbose=False)
            elif 'lucretili' in folder_lower:
                _, _, _, rule_masks = self.engine._apply_lucretili_rules(anomalies.copy(), anomaly_sources, df=df, verbose=False)
            elif 'ranverso' in folder_lower:
                _, _, _, rule_masks = self.engine._apply_ranverso_rules(anomalies.copy(), anomaly_sources, df=df, verbose=False)
            elif 'schenkenberg' in folder_lower:
                _, _, _, rule_masks = self.engine._apply_schenkenberg_rules(anomalies.copy(), anomaly_sources, df=df, verbose=False)
            else:
                raise ValueError("Unknown rule set")
            return rule_masks

        rule_masks_orig = _run_rules(self.raw_df)

        for rule in masks_df.columns:
            mask = masks_df[rule].astype(bool)
            if len(mask) == 0:
                diagnostics[rule] = dict(activation_rate=np.nan, burstiness=np.nan, noise_sensitivity=np.nan)
                continue
            activation_rate = mask.mean()
            idx = np.where(mask)[0]
            if len(idx) < 2:
                burstiness = np.nan
            else:
                inter = np.diff(idx)
                mu = inter.mean()
                sigma = inter.std()
                burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0
            flips_total = 0
            total = 0
            n_trials = 10  # Noise re-runs used to estimate noise_sensitivity
            for _ in range(n_trials):
                noisy_df = self._add_noise(self.raw_df, noise_std)
                rule_masks_noisy = _run_rules(noisy_df)
                mask_orig = rule_masks_orig[rule]
                mask_noisy = rule_masks_noisy[rule]
                valid_mask = mask_orig.notna() & mask_noisy.notna()
                flips_total += np.sum(mask_orig[valid_mask] != mask_noisy[valid_mask])
                total += np.sum(valid_mask)
            noise_sensitivity = flips_total / total if total > 0 else np.nan
            diagnostics[rule] = dict(activation_rate=activation_rate, burstiness=burstiness, noise_sensitivity=noise_sensitivity)

        df_diag = pd.DataFrame(diagnostics).T

        return df_diag

    def _add_noise(self, df, noise_std=0.05):
        """Add zero-mean Gaussian noise scaled to each column's own std
        (noise_std as a fraction of that std). Used only for the
        noise_sensitivity robustness check above."""
        noisy = df.copy()
        for col in noisy.columns:
            v = pd.to_numeric(noisy[col], errors='coerce')
            if v.notna().sum() == 0:
                continue
            scale = np.nanstd(v)
            if scale == 0 or np.isnan(scale):
                continue
            noise = np.random.normal(0, noise_std * scale, size=len(v))
            noisy[col] = v + noise
        return noisy


class TSDataset(Dataset):
    '''
    Dataset for self-supervised reconstruction + forecasting.
    - Loads raw data with alive masks.
    - Places expert knowledge (EK) anomalies in test set.
    - Splits into train / val / test (temporal).
    - Imputes missing values and scales with StandardScaler (saved to disk).
    - Optional augmentation for training.
    - Injects synthetic anomalies into test set for evaluation.
    '''

    def __init__(self, mode, data, feature_names, lb, val_ratio, test_ratio,
                 split_order=None, positive=None, augment=False, figure_path="",
                 scaler_save_path=None, inject_on_init=True, anom_freq=0.01, anom_type="all",
                 anom_sev=1, natural_extreme_c=12.0, natural_extreme_window=48, 
                 train_history_pct=None,  baseline_preprocessing=False):

        self.learning_mode = mode
        self.train_history_pct = train_history_pct   # <-- NEW

        full_data = data[feature_names].copy()

        # --- NEW: trim BEFORE any split is computed ---
        if train_history_pct is not None:
            if not (0 < train_history_pct <= 1.0):
                raise ValueError("train_history_pct must be in (0, 1].")
            n_total_full = len(full_data)
            n_keep = int(round(train_history_pct * n_total_full))
            n_keep = max(n_keep, 1)
            full_data = full_data.iloc[-n_keep:]   # most recent n_keep rows only
            print(f"[train_history_pct={train_history_pct}] "
                f"trimmed dataset to {len(full_data)}/{n_total_full} rows "
                f"({len(full_data)/n_total_full:.1%})")
        # --- end new ---

        self.data = full_data
        self.feature_names = feature_names
        self.val_ratio = val_ratio
        self.lb = lb
        self.inject_on_init = inject_on_init
        self.test_ratio = test_ratio
        self.split_order = split_order
        self.ek_rules = positive.fillna(0.0) if positive is not None else None
        self.use_augment = augment
        self.figure_path = figure_path
        self.scaler_save_path = scaler_save_path  # path to save/load fitted scalers
        self.anom_freq = anom_freq
        self.anom_type = anom_type
        self.anom_sev = anom_sev
        self.natural_extreme_c = natural_extreme_c
        self.natural_extreme_window = natural_extreme_window
        self.train_history_pct = train_history_pct 
        self.baseline_preprocessing = baseline_preprocessing
        if self.baseline_preprocessing:
           print ("Using same preprocessing as in baselines: No Fc, no alive mask, no revin")
        
        # NOTE: assumes exactly 6 trailing cyclic columns (hour/dow/doy sin+cos)
        # appended after the raw+alive feature blocks -- see _preprocess_data.
        self.n_features_base = (data.shape[1] - 6) // 2
        self.n_features_ext = (data.shape[1] + 6) // 2

        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

        self._preprocess_data()

        self.mode = 'train'
        self.current_data = self.all_data[self.mode]

    def __len__(self):
        # -lb-1 so that __getitem__ can safely access idx+lb (forecast target)
        return max(0, len(self.current_data) - self.lb - 1)

    def __getitem__(self, idx):
        """
        Returns (x, y_rec, y_fc, alive, ek_label).
        x       : [lb, n_features_ext] -- input window (no forecast step)
        y_rec   : [lb, n_features]      -- reconstruction target (same window, no cyclic)
        y_fc    : [n_features]          -- forecast target (next timestep after window)
        alive   : [lb+1, n_features]    -- alive mask for the full window+1
        ek_label: scalar float           -- 1 if any EK anomaly in window
        """
        full_window = self.current_data[idx: idx + self.lb + 1]
        n_features = (full_window.shape[1] - 6) // 2
        n_features_ext = (full_window.shape[1] + 6) // 2
        alive = full_window[:, -n_features:]  # [lb+1, n_features]

        if self.mode == "test":
            raw_levels = self.test_raw[idx: idx + self.lb + 1]
            x = raw_levels[:-1, :]
            y_rec = raw_levels[:-1, :n_features]
            y_fc = raw_levels[-1, :n_features]
        else:
            levels = full_window[:, :n_features_ext]
            if self.use_augment and self.mode == "train":
                levels = self.augment(levels)
            raw_levels = levels
            x = levels[:-1, :]
            y_rec = raw_levels[:-1, :n_features]
            y_fc = raw_levels[-1, :n_features]
        
        # ---------------------------------------------------------
        # Baseline preprocessing:
        #   1. Ignore sensor availability information
        #   2. Remove temporal/context features from the input
        # ---------------------------------------------------------
        if self.baseline_preprocessing:
            alive = np.ones_like(alive)

            # Last six columns of x are cyclic/context features
            x = x.copy()
            x[:, -6:] = 0.0

        if self.ek_splits is not None:
            ek_label = int(np.any(self.ek_splits[self.mode][idx: idx + self.lb]))
        else:
            ek_label = 0

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_rec, dtype=torch.float32),
            torch.tensor(y_fc, dtype=torch.float32),
            torch.tensor(alive, dtype=torch.float32),
            torch.tensor(ek_label, dtype=torch.float32),
        )

    def _preprocess_data(self):
        """
        Full preprocessing pipeline: numeric coercion -> temporal train/val/test
        split -> drop low-alive features (judged per split) -> impute+scale
        (fit on train only) -> optionally inject synthetic test anomalies ->
        optionally save inspection plots.
        """
        total = len(self.data)

        alive_cols = [c for c in self.feature_names if c.endswith("_alive")]
        cyclic_cols = [c for c in self.feature_names if c in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos')]
        feature_cols = [c for c in self.feature_names if c not in alive_cols and c not in cyclic_cols]

        self.n_features_base = len(feature_cols)

        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Extract as numpy with columns ordered as [features, cyclic] and [alive] separately
        raw_values = self.data[feature_cols + cyclic_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        alive_mask = self.data[alive_cols].to_numpy(dtype=np.float64, na_value=np.nan)

        all_idx = np.arange(total)
        n_test = int(self.test_ratio * total)
        n_val = int(self.val_ratio * total)
        n_train = total - n_val - n_test

        # Temporal split (no shuffling, since these are ordered time series)
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train: n_train + n_val]
        test_idx = all_idx[n_train + n_val:]

        # Drop features with insufficient alive ratio in any split
        kept_features, kept_features_core = self.drop_low_alive_features(train_idx, val_idx, test_idx, alive_mask, raw_values, min_ratio=0.10)
        raw_values = raw_values[:, kept_features]
        alive_mask = alive_mask[:, kept_features_core]

        self.feature_names = [self.feature_names[i] for i in kept_features]

        train_raw = raw_values[train_idx]
        val_raw = raw_values[val_idx]
        test_raw = raw_values[test_idx]
        train_alive = alive_mask[train_idx]
        val_alive = alive_mask[val_idx]
        test_alive = alive_mask[test_idx]

        # Impute + scale (StandardScaler, fit on train only, saved to disk)
        train_scaled, val_scaled, test_scaled = self.preprocess_split(
            train_raw, val_raw, test_raw, train_alive)

        train_data = np.concatenate([train_scaled, train_alive], axis=1)
        val_data = np.concatenate([val_scaled, val_alive], axis=1)
        test_data = np.concatenate([test_scaled, test_alive], axis=1)
        print("train, val, test data: ", train_data.shape, val_data.shape, test_data.shape)

        self.all_data = {"train": train_data, "val": val_data, "test": test_data}

        self.test_ek_mask = np.zeros(len(test_idx), dtype=int)
        if self.ek_rules is not None:
            self.test_ek_mask = self.ek_rules[test_idx]

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        if self.ek_rules is not None:
            ek_int = (np.array(self.ek_rules) > 0.5).astype(int)
            self.ek_splits = {
                "train": ek_int[train_idx],
                "val": ek_int[val_idx],
                "test": ek_int[test_idx],
            }
        else:
            self.ek_splits = None

        if self.inject_on_init:
            self.inject_test_anomalies(freq=self.anom_freq, feature_names=self.feature_names[:-6], anom_type=self.anom_type, sev=self.anom_sev)
        else:
            # No synthetic injection: test_raw/test_clean are just the clean  (real) test split, so downstream eval code always has these
            # attributes available regardless of inject_on_init.
            n_test_features = (self.all_data['test'].shape[1] - 6) // 2
            n_test_features_ext = (self.all_data['test'].shape[1] + 6) // 2
            self.test_clean = self.all_data["test"][:, :n_test_features_ext].copy()
            self.test_raw = self.test_clean.copy()
            self.test_scaled_clean = self.all_data["test"].copy()
            reference_values = np.concatenate([self.all_data["train"], self.all_data["val"]], axis=0)[:, :n_test_features]
            alive_reference_values = np.concatenate([self.all_data["train"], self.all_data["val"]], axis=0)[:, -n_test_features:]
            feature_scale = self._feature_scale(reference_values, alive_reference_values)

        if self.figure_path and self.figure_path.strip():
            hp.feature_input_model_plot(
                train_data=self.all_data["train"],
                val_data=self.all_data["val"],
                test_data=self.all_data["test"],
                feature_names=self.feature_names,
                figure_path=self.figure_path,
            )

    def preprocess_split(self, train_raw, val_raw, test_raw, alive_train,
                          clip_percentile=99):
        """
        Impute (mean, fit on train) -> per-feature StandardScaler, fit only
        on ALIVE train values after winsorising to [100-clip_percentile,
        clip_percentile] to avoid outlier-driven scale -> transform all
        splits. Scaler instances (+ imputer) saved to self.scaler_save_path
        if provided, so inference can reuse the exact train-time fit.

        """
        train_imp = self.imputer.fit_transform(train_raw)
        if len(val_raw) > 0: 
            val_imp = self.imputer.transform(val_raw)
        else: 
            val_imp = val_raw.copy()
        if len(test_raw) > 0: 
            test_imp = self.imputer.transform(test_raw)
        else: 
            test_imp=test_raw.copy()

        n_features = train_imp.shape[1] - 6  # exclude cyclic
        n_features_ext = train_imp.shape[1]

        train_scaled = train_imp.copy()
        val_scaled = val_imp.copy()
        test_scaled = test_imp.copy()

        fitted_scalers = {}

        for f in range(n_features):
            alive_idx = np.where(alive_train[:, f] == 1)[0]

            if len(alive_idx) == 0:
                # No alive values: skip scaling for this feature
                fitted_scalers[f] = None
                continue

            alive_vals = train_imp[alive_idx, f]

            # Winsorise fit range to avoid outlier-driven scale
            p_low = np.percentile(alive_vals, 100 - clip_percentile)
            p_high = np.percentile(alive_vals, clip_percentile)

            if p_high - p_low < 1e-8:
                # Near-constant feature: no scaling needed
                fitted_scalers[f] = None
                continue

            scaler_f = copy.deepcopy(self.scaler)  # deepcopy preserves any constructor kwargs on self.scaler
            clipped_alive = np.clip(alive_vals, p_low, p_high)
            scaler_f.fit(clipped_alive.reshape(-1, 1))

            # Guard against zero scale (e.g. all-same values after clip)
            if hasattr(scaler_f, "scale_") and np.any(scaler_f.scale_ == 0):
                scaler_f.scale_[scaler_f.scale_ == 0] = 1.0

            train_scaled[:, f] = scaler_f.transform(train_imp[:, f].reshape(-1, 1)).flatten()
            if len(val_raw) > 0: 
                val_scaled[:, f] = scaler_f.transform(val_imp[:, f].reshape(-1, 1)).flatten()
            if len(test_raw) > 0: 
                test_scaled[:, f] = scaler_f.transform(test_imp[:, f].reshape(-1, 1)).flatten()

            fitted_scalers[f] = scaler_f

        if self.scaler_save_path:
            os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
            with open(self.scaler_save_path, "wb") as f:
                pickle.dump({
                    "scalers": fitted_scalers,
                    "imputer": self.imputer,
                    "n_features": n_features,
                }, f)
            #print(f"Scalers saved to {self.scaler_save_path}")

        self.fitted_scalers = fitted_scalers
        return train_scaled, val_scaled, test_scaled

    def drop_low_alive_features(self, train_idx, val_idx, test_idx, alive_mask, data, min_ratio=0.10):
        idx_splits = [train_idx, val_idx, test_idx]
        n_valid_features = alive_mask.shape[1]
        n_total_features = data.shape[1]
        keep_mask = np.ones(n_valid_features, dtype=bool)
        data_valid = data[:, :n_valid_features]

        for split_name, idx in zip(["train", "val", "test"], idx_splits):
            if len(idx) == 0:
                continue  # <-- ADDED: nothing to check on an empty split

            alive_counts = alive_mask[idx].sum(axis=0)
            required = np.ceil(min_ratio * len(idx))
            low_alive = alive_counts < required

            df_check = pd.DataFrame(data_valid[idx])
            all_nan = df_check.isna().all(axis=0).values

            keep_mask &= ~(low_alive | all_nan)

        kept_features_core = np.where(keep_mask)[0]
        last_6_features = np.arange(n_valid_features, n_total_features)
        kept_features = np.concatenate([kept_features_core, last_6_features])
        #print(f"Core kept: {len(kept_features_core)} / {n_valid_features}")
        self.n_features = len(kept_features)
        return kept_features, kept_features_core

    def _check_split_health(self, split_raw, name):
        """Diagnostic helper: print any feature whose alive ratio in this
        split is below 5%. NOTE: not called anywhere in this file --
        confirm it's still used elsewhere (e.g. a notebook) before removing.
        """
        n_features = (split_raw.shape[1] - 6) // 2
        alive_masks = split_raw[:, -n_features:].astype(float)
        for i, feat in enumerate(self.feature_names):
            alive_ratio = alive_masks[:, i].sum() / len(alive_masks)
            if alive_ratio < 0.05:
                print(f"--- {feat} --- Alive ratio: {alive_ratio:.2%}")

    def clean_numeric(self, x):
        """Coerce pandas NA / object-dtype arrays to a clean float64 numpy
        array. NOTE: not called anywhere in this file -- confirm it's still
        used elsewhere before removing."""
        if isinstance(x, pd.DataFrame):
            x = x.replace({pd.NA: np.nan})
            return x.to_numpy(dtype=np.float64)

        x = np.asarray(x)

        if x.dtype == object:
            x = np.where(x == pd.NA, np.nan, x)
            x = x.astype(np.float64)

        return x

    def augment(self, x):
        """Lightweight training-time augmentation: random per-feature scale
        jitter and additive Gaussian noise, each applied with 25% probability."""
        x = x.copy()
        if np.random.rand() < 0.25:  
            factor = np.random.uniform(0.9, 1.1, size=(1, x.shape[1]))
            x *= factor
        if np.random.rand() < 0.25:
            noise = np.random.normal(0, 0.01 * np.std(x, axis=0), size=x.shape)
            x += noise
        return x

    def detect_periodic_features_multi(self, values, period_len, threshold=0.5):
        """
        Identify which features are periodic with period `period_len`, via
        autocorrelation at lag=period_len that's both above `threshold` and
        clearly higher than at half/near-period lags (to avoid flagging
        features whose autocorrelation decays smoothly rather than peaking).
        Used to target contextual anomaly injection at genuine peaks/troughs.
        """
        N, F = values.shape
        periodic_features = []
        for f in range(F):
            if N <= period_len:
                continue
            x = values[:, f]
            if np.std(x) < 1e-6:
                continue

            def autocorr(lag):
                if lag <= 0 or lag >= N:
                    return 0
                x1 = x[:-lag]
                x2 = x[lag:]
                if np.std(x1) < 1e-6 or np.std(x2) < 1e-6:
                    return 0
                return np.corrcoef(x1, x2)[0, 1]

            acf_p = autocorr(period_len)
            acf_half = autocorr(period_len // 2)
            acf_near = max(autocorr(period_len - period_len // 6),
                           autocorr(period_len + period_len // 6))

            # Margins (0.20, 0.05) chosen empirically to require a clear
            # peak at the true period vs. neighbouring lags
            if acf_p > threshold and acf_p > acf_half + 0.20 and acf_p > acf_near + 0.05:
                periodic_features.append(f)
        return periodic_features

    def find_peaks_troughs(self, x, period_len):
        """Smooth the series and locate its periodic peaks/troughs (used as
        donor/target locations for contextual anomaly injection)."""
        smooth = gaussian_filter1d(x, sigma=max(1, period_len // 8))
        prom = np.std(smooth) * 0.4  # Prominence threshold chosen empirically
        peaks, _ = find_peaks(smooth, distance=period_len // 2, prominence=prom)
        troughs, _ = find_peaks(-smooth, distance=period_len // 2, prominence=prom)
        return peaks, troughs

    def build_corr_matrix(self, values, valid_features):
        """Pairwise absolute Pearson correlation matrix over valid_features,
        used to find correlated 'partner' features for multivariate
        (co-injected) synthetic anomalies."""
        sub = values[:, valid_features].astype(float)
        n = len(valid_features)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rho = np.corrcoef(sub[:, i], sub[:, j])[0, 1]
                mat[i, j] = mat[j, i] = abs(rho) if np.isfinite(rho) else 0.0
        return mat

    def _feature_scale(self, values_ref, alive_ref, min_scale=1e-8):
        """
        Per-feature magnitude scale = 1st-99th percentile range of alive
        reference values (train+val). Used as the injection magnitude unit
        (severity * scale), so a flat sensor (e.g. a counter) still gets a
        meaningful injection instead of a near-zero one.
        """
        F = values_ref.shape[1]
        scale = np.full(F, min_scale)
        for f in range(F):
            xf = values_ref[:, f]
            mf = alive_ref[:, f]
            xf_alive = xf[mf == 1]
            if len(xf_alive) == 0:
                scale[f] = min_scale
                continue
            q1, q99 = np.percentile(xf_alive, 1), np.percentile(xf_alive, 99)
            s = q99 - q1
            scale[f] = max(s, min_scale)
        self.feat_scale = scale
        return scale

    def inject_test_anomalies(self, freq=0.01, period_len=48, sev=6.0, cooldown_factor=1.0,
                               type_ratios=None, n_instances=None, corr_thresh=0.75, max_partners=2,
                               anom_type="all", min_std=1e-6, feature_names=None, scale_pct=95.0):
        """
        Inject synthetic collective / contextual / point anomalies into the
        (clean) test split, round-robin across the requested types so no
        single type starves the others. Anomaly magnitude is `sev *
        feature_scale` (see _feature_scale), and correlated 'partner'
        features are co-injected to create realistic multivariate anomalies.
        Populates self.test_raw / self.injected_anomalies_labels /
        self.{collective,contextual,point}_idx and optionally saves
        before/after inspection plots per injected instance.
        """
        print("Using severity:", sev)

        test_data = self.all_data["test"].copy()
        n_features = (test_data.shape[1] - 6) // 2
        n_features_ext = (test_data.shape[1] + 6) // 2

        original_values = test_data[:, :n_features].copy()
        values = original_values.copy()
        N = values.shape[0]
        start_valid, end_valid = self.lb, N - self.lb
        cooldown_min = int(self.lb * 1.5) # 108
        if freq==0.10: 
            cooldown_min=cooldown_min//2 #54
        elif freq==0.2: 
            cooldown_min=cooldown_min//4 #27
        print("Cooldown", cooldown_min)
            

        feat_std = np.std(values, axis=0)
        abs_ok = feat_std > min_std
        rel_ok = feat_std > np.percentile(feat_std[abs_ok], 10) if abs_ok.any() else abs_ok  # bottom-10th-percentile-std features treated as effectively constant
        valid_features = np.where(abs_ok & rel_ok)[0]
        excluded = np.where(~(abs_ok & rel_ok))[0]
        if len(excluded) > 0:
            print(f"Excluded constant/near-constant features: {list(excluded)}")

        # Global per-feature scale computed from TRAIN+VAL only (never test), so injection magnitude doesn't leak information from the test split.
        reference_values = np.concatenate([self.all_data["train"], self.all_data["val"]], axis=0)[:, :n_features]
        alive_reference_values = np.concatenate([self.all_data["train"], self.all_data["val"]], axis=0)[:, -n_features:]
        feature_scale = self._feature_scale(reference_values, alive_reference_values)

        corr_mat = self.build_corr_matrix(values, valid_features)
        feat_to_idx = {f: i for i, f in enumerate(valid_features)}

        def get_partners(f):
            """Up to max_partners features correlated with f above corr_thresh."""
            if f not in feat_to_idx:
                return []
            row = corr_mat[feat_to_idx[f]].copy()
            row[feat_to_idx[f]] = 0.0
            strong = np.where(row >= corr_thresh)[0]
            strong = strong[np.argsort(-row[strong])][:max_partners]
            return [valid_features[j] for j in strong]

        def make_mode_pool(modes):
            """Shuffled deque of injection sub-modes, refilled+reshuffled
            (see next_mode) once exhausted, to keep sub-mode usage balanced."""
            pool = list(modes)
            random.shuffle(pool)
            return deque(pool)

        def next_mode(pool, modes):
            if not pool:
                refill = list(modes)
                random.shuffle(refill)
                pool.extend(refill)
            return pool.popleft()

        periodic_candidates = self.detect_periodic_features_multi(values[:, valid_features], period_len)
        periodic_features = valid_features[periodic_candidates]
        extrema = {}
        for f in periodic_features:
            pk, tr = self.find_peaks_troughs(values[start_valid:end_valid, f], period_len)
            extrema[f] = (pk + start_valid, tr + start_valid)

        occupied = np.zeros(N, dtype=bool)
        self.anomaly_indices, anomaly_ranges = [], []
        idx_list = {"collective": [], "contextual": [], "point": []}

        def is_free(s, e):
            s, e = max(0, s), min(N, e)
            return not occupied[s:e].any()

        def mark(s, e, seg_len):
            """Mark [s-cd, e+cd] as occupied so subsequent anomalies don't
            overlap or sit immediately next to this one."""
            cd = max(int(seg_len * cooldown_factor), cooldown_min)
            occupied[max(0, s - cd):min(N, e + cd)] = True

        def register(tag, s, e, features, mode, multi):
            self.anomaly_indices.append({
                "type": tag, "mode": mode, "start": int(s), "end": int(e),
                "center": int((s + e) // 2), "features": tuple(int(x) for x in features),
                "sev": sev, "multi": multi,
            })
            anomaly_ranges.append((s, e))
            idx_list[tag].append(self.anomaly_indices[-1])

        len_collective = 12 
        num_windows = N - self.lb - 1
        max_windows = max(1, int(freq * num_windows))

        collective_mode = None

        if anom_type == "all":
            active_types = ["collective", "contextual", "point"]
        elif anom_type == "point":
            active_types = ["point"]
        elif anom_type == "contextual":
            active_types = ["contextual"]
        elif anom_type == "collective_all":
            active_types = ["collective"]
        elif anom_type == "collective_shift":
            active_types = ["collective"]
            collective_mode = "shift"
        elif anom_type == "collective_variance":
            active_types = ["collective"]
            collective_mode = "variance_burst"
        elif anom_type == "collective_frozen":
            active_types = ["collective"]
            collective_mode = "frozen_sensor"
        else:
            raise ValueError(f"Unknown anomaly type: {anom_type}")

        if n_instances is not None:
            counts = {t: n_instances.get(t, 0)for t in ["collective", "contextual", "point"]}
        else:
            if type_ratios is None:
                type_ratios = {t: 1.0 for t in active_types}

            type_lengths = {
                "collective": 12,
                "contextual": 3,
                "point": 1,
            }

            occupancy_budget = max_windows
            counts = {t: 0 for t in active_types}

            used = 0

            # Round-robin allocation. Once an anomaly makes the occupancy
            # exceed the frequency budget, stop allocating any more.
            while True:
                added = False

                for t in active_types:
                    if type_ratios.get(t, 0) <= 0:
                        continue

                    length = type_lengths[t]

                    counts[t] += 1
                    used += length
                    added = True

                    # Important: the anomaly that crossed the budget is kept,
                    # but nothing else can be added afterwards.
                    if used >= occupancy_budget:
                        break

                if not added or used >= occupancy_budget:
                    break

            for t in ["collective", "contextual", "point"]:
                counts.setdefault(t, 0)

            print(
                f"max_windows={max_windows} | "
                f"occupancy_budget={occupancy_budget} | "
                f"target instance counts={counts} | "
                f"planned occupancy={used}"
            )

        INNER_TRIES = 20  # Max retries per placement attempt before giving up on that slot

        # ---------------- COLLECTIVE ----------------
        if collective_mode is None:
            modes_coll = ["shift", "variance_burst", "frozen_sensor"]
        else:
            modes_coll = [collective_mode]

        pool_coll = make_mode_pool(modes_coll)

        def try_place_collective():
            """Attempt to place one collective anomaly (shift / variance
            burst / frozen segment) plus co-injection on correlated partners."""
            nonlocal pool_coll
            for _ in range(INNER_TRIES):
                s = np.random.randint(start_valid, end_valid - len_collective)
                e = s + len_collective
                if not is_free(s, e):
                    continue
                f = np.random.choice(valid_features)
                seg = values[s:e, f].copy()
                mode = next_mode(pool_coll, modes_coll)
                mag = sev * feature_scale[f]
                if mode == "shift":
                    values[s:e, f] = seg + np.random.choice([-1, 1]) * mag
                elif mode == "variance_burst":
                    values[s:e, f] = seg + np.random.normal(0, mag, len_collective)
                else:
                    values[s:e, f] = seg[0]

                co_features = []
                for pf in get_partners(f):
                    mag_p = sev * feature_scale[pf]
                    seg_p = values[s:e, pf].copy()
                    if mode == "shift":
                        values[s:e, pf] = seg_p + np.random.choice([-1, 1]) * mag_p
                    elif mode == "variance_burst":
                        values[s:e, pf] = seg_p + np.random.normal(0, mag_p, len_collective)
                    else:
                        values[s:e, pf] = seg_p[0]
                    co_features.append(pf)

                register("collective", s, e, [f] + co_features, mode, multi=len(co_features) > 0)
                mark(s, e, len_collective)
                return True
            return False

        # ---------------- CONTEXTUAL ----------------
        seg_len = 3  # Contextual-anomaly segment length, in samples
        candidates = []
        for f in periodic_features:
            pks, trs = extrema[f]
            candidates += [(f, p, True) for p in pks if start_valid <= p - 1 and p + 2 <= end_valid]
            candidates += [(f, t, False) for t in trs if start_valid <= t - 1 and t + 2 <= end_valid]
        np.random.shuffle(candidates)
        ctx_ptr = 0

        def try_place_contextual():
            """Attempt to place one contextual anomaly: swap a peak segment
            for a trough segment (or vice-versa) at a periodic feature,
            plus a correlation-consistent swap on correlated partners."""
            nonlocal ctx_ptr
            while ctx_ptr < len(candidates):
                f, center, is_peak = candidates[ctx_ptr]
                ctx_ptr += 1
                s, e = center - 1, center + 2
                if not is_free(s, e):
                    continue
                donor_pool = extrema[f][1] if is_peak else extrema[f][0]
                if len(donor_pool) == 0:
                    continue
                donor_center = donor_pool[np.argmin(np.abs(donor_pool - center))]
                if abs(donor_center - center) < seg_len * 2:
                    continue
                ds, de = donor_center - 1, donor_center + 2
                if ds < 0 or de > N:
                    continue
                donor_window = original_values[ds:de, f]
                if len(donor_window) != seg_len or np.corrcoef(donor_window, original_values[s:e, f])[0, 1] > 0.85:
                    continue
                values[s:e, f] = donor_window

                co_features = []
                for pf in get_partners(f):
                    if pf not in extrema:
                        continue
                    donor_pool_p = extrema[pf][1] if is_peak else extrema[pf][0]
                    if len(donor_pool_p) == 0:
                        continue
                    dc_p = donor_pool_p[np.argmin(np.abs(donor_pool_p - center))]
                    if abs(dc_p - center) < seg_len * 2:
                        continue
                    ds_p, de_p = dc_p - 1, dc_p + 2
                    if ds_p < 0 or de_p > N:
                        continue
                    dw_p = original_values[ds_p:de_p, pf]
                    ow_p = original_values[s:e, pf]
                    if len(dw_p) == seg_len and np.corrcoef(dw_p, ow_p)[0, 1] <= 0.85:
                        values[s:e, pf] = dw_p
                        co_features.append(pf)

                register("contextual", s, e, [f] + co_features, "context_swap", multi=len(co_features) > 0)
                mark(s, e, seg_len)
                return True
            return False

        # ---------------- POINT ----------------
        modes_pt = ["spike", "drop"]
        pool_pt = make_mode_pool(modes_pt)

        def try_place_point():
            """Attempt to place one point anomaly (single-sample spike or
            drop) plus co-injection on correlated partners."""
            nonlocal pool_pt
            for _ in range(INNER_TRIES):
                t = np.random.randint(start_valid, end_valid)
                if not is_free(t, t + 1):
                    continue
                f = np.random.choice(valid_features)
                mag = sev * feature_scale[f]
                mode = next_mode(pool_pt, modes_pt)
                ref = values[t - 1, f]
                values[t, f] = ref + mag if mode == "spike" else ref - mag

                co_features = []
                for pf in get_partners(f):
                    mag_p = sev * feature_scale[pf]
                    ref_p = values[t - 1, pf]
                    values[t, pf] = ref_p + mag_p if mode == "spike" else ref_p - mag_p
                    co_features.append(pf)

                register("point", t, t + 1, [f] + co_features, mode, multi=len(co_features) > 0)
                mark(t, t + 1, 1)
                return True
            return False

        # ---------------- ROUND-ROBIN DRIVER ----------------
        # Cycles through active anomaly types placing one instance at a time,
        # instead of filling one type completely before moving to the next --
        # this is what prevents e.g. point anomalies from starving when
        # collective/contextual placements are easier to find slots for.
        placers = [("collective", try_place_collective),
                   ("contextual", try_place_contextual),
                   ("point", try_place_point)]
        placers = [(t, fn) for t, fn in placers if counts.get(t, 0) > 0]

        c_counts = {"collective": 0, "contextual": 0, "point": 0}
        fail_streak = {"collective": 0, "contextual": 0, "point": 0}
        MAX_FAIL_STREAK = 2000  # Give up on a type after this many consecutive failed placement attempts

        while placers:
            still_active = []
            for tag, fn in placers:
                if c_counts[tag] >= counts[tag]:
                    continue
                if fail_streak[tag] >= MAX_FAIL_STREAK:
                    continue
                if fn():
                    c_counts[tag] += 1
                    fail_streak[tag] = 0
                else:
                    fail_streak[tag] += 1
                if c_counts[tag] < counts[tag] and fail_streak[tag] < MAX_FAIL_STREAK:
                    still_active.append((tag, fn))
            placers = still_active

        # ---------------- OUTPUT ----------------
        full_test = test_data.copy()
        full_test[:, :n_features] = values
        self.test_raw = full_test[:, :n_features_ext]
        self.test_clean = test_data[:, :n_features_ext].copy()
        self.collective_idx = idx_list["collective"]
        self.contextual_idx = idx_list["contextual"]
        self.point_idx = idx_list["point"]
        self.all_data["test"][:, :n_features] = values

        inj_labels = np.zeros(num_windows, dtype=int)
        for s, e in anomaly_ranges:
            start_w = max(0, s - self.lb)
            end_w = min(num_windows, e - self.lb + 1)
            inj_labels[start_w:end_w] = 1
        self.injected_anomalies_labels = inj_labels

        print(f"Injected anomalies: {inj_labels.sum()} windows | actual counts: "
              f"collective={len(idx_list['collective'])}, "
              f"contextual={len(idx_list['contextual'])}, "
              f"point={len(idx_list['point'])}")

        plot_path = self.figure_path.replace("split_plot", "injected_plots")
        if self.figure_path and self.figure_path.strip():
            if os.path.exists(plot_path):
                shutil.rmtree(plot_path)
            os.makedirs(plot_path, exist_ok=True)
            for entry in self.anomaly_indices:
                hp.plot_injected(
                    self.test_clean, self.test_raw,
                    entry["center"], entry["features"],
                    f"{entry['type']}_{entry['mode']}",
                    self.feature_names, plot_path, lookback=self.lb
                )
