
import numpy as np
import pandas as pd


class RuleEngine:
    """Unified, consistent, robust rule engine for all 5 pilots."""

    def __init__(self, raw_df, threshold_overrides=None, freq="5min"):
        self.raw_df = raw_df.sort_index()
        self.threshold_overrides = threshold_overrides or {}
        self._printed_thresholds = set()

        self._freq = pd.Timedelta(freq)
        self._steps = self._make_steps(self._freq)
        
        self._printed_triggers = set()

    # HELPERS
    def _make_steps(self, freq):
        def steps(duration_str):
            return max(1, int(pd.Timedelta(duration_str) / freq))
        return steps

    @staticmethod
    def _sigmoid_score(x, threshold, k=1.0):
        x = pd.Series(x) if not isinstance(x, pd.Series) else x
        diff = x - threshold
        scale = abs(threshold) if abs(threshold) > 1e-8 else float(x.std() + 1e-8)
        score =1.0 / (1.0 + np.exp(-k * diff / scale))
        return score

    def _thr(self, key, default, verbose=True):
        val = self.threshold_overrides.get(key, default)
        if verbose and key not in self._printed_thresholds:
            #print(f"[THR] {key} = {val} {'(override)' if key in self.threshold_overrides else '(default)'}")
            self._printed_thresholds.add(key)
        return val

    def _empty(self):
        idx = self.raw_df.index
        return (pd.Series(0.0, index=idx),pd.Series(0.0, index=idx),pd.Series(False, index=idx),)

    def _align(self, s):
        """Align any Series to the main index."""
        s = pd.Series(s)
        s = s.reindex(self.raw_df.index)
        s = s.ffill().fillna(0)
        return s

    def _max_aligned(self, *series):
        """Elementwise max with automatic alignment."""
        return pd.concat(series, axis=1).max(axis=1)

    def _apply_rule(self, rule_score, rule_raw, rule_mask, metric, thr, col, anomalies, anomaly_sources):
  
        metric = self._align(metric)
        score = self._sigmoid_score(metric, thr)
        score = pd.Series(score, index=self.raw_df.index).fillna(0.0)

        mask_internal = self._align(metric > thr).astype(bool)

        rule_score = self._max_aligned(rule_score, score)
        rule_raw = self._max_aligned(rule_raw, metric)
        rule_mask |= mask_internal

        anomalies |= mask_internal
        anomaly_sources[col] |= mask_internal

        if not hasattr(self, "_printed_vars"):
            self._printed_vars = set()

        if col not in self._printed_vars:
            self._printed_vars.add(col)
        return rule_score, rule_raw, rule_mask, anomalies, anomaly_sources

    # =======================  DELOS  =============================
    def _apply_delos_rules(self, anomalies, anomaly_sources, df=None, verbose=True):
        if df is None:
            df = self.raw_df

        df = df.sort_index()
        rule_scores, metrics, rule_masks = {}, {}, {}

        # ---------------- R1: Rain 1h / 3h ----------------
        r1a, r1a_raw, r1a_mask = self._empty()   # 1h
        r1b, r1b_raw, r1b_mask = self._empty()   # 3h

        rain_cols = [c for c in df.columns if 'rain' in c.lower() and 'acc' not in c.lower()]

        w1h = self._steps('1h')
        w3h = self._steps('3h')

        for col in rain_cols:
            v = pd.to_numeric(df[col], errors='coerce')

            # --- 1h window ---
            roll_1h = v.rolling(w1h, min_periods=1).sum()
            thr1 = self._thr('R1a_rain_1h', 10, verbose)

            r1a, r1a_raw, r1a_mask, anomalies, anomaly_sources = \
                self._apply_rule(r1a, r1a_raw, r1a_mask, roll_1h, thr1, col, anomalies, anomaly_sources)

            # --- 3h window ---
            roll_3h = v.rolling(w3h, min_periods=1).sum()
            thr3 = self._thr('R1b_rain_3h', 60, verbose)

            r1b, r1b_raw, r1b_mask, anomalies, anomaly_sources = \
                self._apply_rule(r1b, r1b_raw, r1b_mask, roll_3h, thr3, col, anomalies, anomaly_sources)

        rule_scores['R1a_rain_1h'] = r1a
        rule_scores['R1b_rain_3h'] = r1b
        metrics['R1a_rain_1h'] = r1a_raw
        metrics['R1b_rain_3h'] = r1b_raw
        rule_masks['R1a_rain_1h'] = r1a_mask
        rule_masks['R1b_rain_3h'] = r1b_mask

        # ---------------- R2: Wind ----------------
        s, m, mask_all = self._empty()
        wind_cols = [c for c in df.columns if 'wind_speed' in c.lower() and 'dir' not in c.lower()]

        for col in wind_cols:
            v = pd.to_numeric(df[col], errors='coerce')
            thr = self._thr('R2_wind', 20, verbose)

            s, m, mask_all, anomalies, anomaly_sources = \
                self._apply_rule(s, m, mask_all, v, thr, col, anomalies, anomaly_sources)

        rule_scores['R2_wind'] = s
        metrics['R2_wind'] = m
        rule_masks['R2_wind'] = mask_all

        # ---------------- R3: Humidity crossings ----------------
        s, m, mask_all = self._empty()
        hum_cols = [c for c in df.columns if 'hum' in c.lower()]

        w12h = self._steps('12h')
        w3d = self._steps('3D')

        for col in hum_cols:
            v_raw = pd.to_numeric(df[col], errors='coerce')
            v = (v_raw
                .interpolate(limit_direction="both")      # rellena huecos sin saltos
                .rolling(w12h, min_periods=1).mean()      # suaviza 12h
                .rolling(3, min_periods=1).mean()         # suaviza aún más (filtro 3 puntos)
            )

            th = self._thr('R3_hum_th', 75, verbose)
            cross_thr = self._thr('R3_cross', 10, verbose)

            cross = (((v.shift(1) < th) & (v > th)) |((v.shift(1) > th) & (v < th)))
            total = cross.rolling(w3d, min_periods=1).sum()

            s, m, mask_all, anomalies, anomaly_sources = \
                self._apply_rule(s, m, mask_all, total, cross_thr, col, anomalies, anomaly_sources)

        rule_scores['R3_hum_cross'] = s
        metrics['R3_hum_cross'] = m
        rule_masks['R3_hum_cross'] = mask_all

        # ---------------- R4: Forced dates ----------------
        forced_mask = pd.Series(False, index=df.index)
        date_index = pd.to_datetime(df.index)

        for d in ["2025-05-30", "2025-12-09"]:
            day_mask = (date_index.date == pd.Timestamp(d).date())

            for col in df.columns:
                v = pd.to_numeric(df[col], errors='coerce')
                day = v[day_mask]
                if day.empty:
                    continue

                p01 = day.quantile(0.01)
                p99 = day.quantile(0.99)
                mask = ((v < p01) | (v > p99)) & day_mask

                forced_mask |= mask
                anomalies |= mask
                anomaly_sources[col] |= mask

        rule_scores['R4_dates'] = forced_mask.astype(float)
        metrics['R4_dates'] = forced_mask.astype(float)
        rule_masks['R4_dates'] = forced_mask

        return anomalies, rule_scores, metrics, rule_masks


    # =======================  BALTANÁS  ==========================
    def _apply_baltanas_rules(self, anomalies, anomaly_sources, df=None, verbose=True):
        if df is None:
            df = self.raw_df

        df = df.sort_index()
        rule_scores, metrics, rule_masks = {}, {}, {}

        # ---------------- R1: PPV ----------------
        r1, r1_raw, r1_mask = self._empty()
        vib_cols = [c for c in df.columns if 'acc_int_ppv' in c.lower()]

        for col in vib_cols:
            v = pd.to_numeric(df[col], errors='coerce')
            thr = self._thr('R1_ppv', 0.25, verbose)

            r1, r1_raw, r1_mask, anomalies, anomaly_sources = \
                self._apply_rule(r1, r1_raw, r1_mask, v, thr, col, anomalies, anomaly_sources)
                
        rule_scores['R1_ppv'] = r1
        metrics['R1_ppv'] = r1_raw
        rule_masks['R1_ppv'] = r1_mask


        # ---------------- R2: Daily humidity range ----------------
        r2, r2_raw, r2_mask = self._empty()
        hum_cols = [c for c in df.columns if 'hum' in c.lower() and 'ee3' not in c.lower() and "hum1" not in c.lower()]

        for col in hum_cols:
            v = pd.to_numeric(df[col], errors='coerce')
            daily_range = (v.resample("1D").max() - v.resample("1D").min()).abs()

            thr = self._thr('R2_hum_daily_range', 5, verbose)

            r2, r2_raw, r2_mask, anomalies, anomaly_sources = \
                self._apply_rule(r2, r2_raw, r2_mask, daily_range, thr, col, anomalies, anomaly_sources)

        rule_scores['R2_hum_daily_range'] = r2
        metrics['R2_hum_daily_range'] = r2_raw
        rule_masks['R2_hum_daily_range'] = r2_mask


        # ---------------- R3: GNSS long-term ----------------
        r3, r3_raw, r3_mask = self._empty()
        gnss_cols = [c for c in df.columns if 'long_term' in c.lower()]

        for col in gnss_cols:
            v = pd.to_numeric(df[col], errors='coerce')
            daily_mean = v.resample("1D").mean()
            daily_change = daily_mean.diff().abs()

            thr = self._thr('R3_gnss_change', 3, verbose)

            r3, r3_raw, r3_mask, anomalies, anomaly_sources = \
                self._apply_rule(r3, r3_raw, r3_mask, daily_change, thr, col, anomalies, anomaly_sources)

        rule_scores['R3_gnss_change'] = r3
        metrics['R3_gnss_change'] = r3_raw
        rule_masks['R3_gnss_change'] = r3_mask


        # ---------------- R4: Soil moisture ----------------
        r4a, r4a_raw, r4a_mask = self._empty()
        r4b, r4b_raw, r4b_mask = self._empty()
        soil_cols = [c for c in df.columns if 'soil_moisture' in c.lower()]

        for col in soil_cols:
            v = pd.to_numeric(df[col], errors='coerce')

            # Low
            thr_low = self._thr('R4a_soil_high', 1.15, verbose)
            r4a, r4a_raw, r4a_mask, anomalies, anomaly_sources = \
                self._apply_rule(r4a, r4a_raw, r4a_mask, v, thr_low, col, anomalies, anomaly_sources)

            # Range
            daily_range = (v.resample("1D").max() - v.resample("1D").min()).abs()
            thr_rng = self._thr('R4b_soil_daily_range', 0.15, verbose)

            r4b, r4b_raw, r4b_mask, anomalies, anomaly_sources = \
                self._apply_rule(r4b, r4b_raw, r4b_mask, daily_range, thr_rng, col, anomalies, anomaly_sources)

        rule_scores['R4a_soil_high'] = r4a
        rule_scores['R4b_soil_range'] = r4b
        metrics['R4a_soil_high'] = r4a_raw
        metrics['R4b_soil_range'] = r4b_raw
        rule_masks['R4a_soil_high'] = r4a_mask
        rule_masks['R4b_soil_range'] = r4b_mask

        # ---------------- R5: Airflow ----------------
        r5a, r5a_raw, r5a_mask = self._empty()
        r5b, r5b_raw, r5b_mask = self._empty()
        airflow_cols = [c for c in df.columns if 'airflow' in c.lower() or 'af' in c.lower()]

        for col in airflow_cols:
            v = pd.to_numeric(df[col], errors='coerce')

            # --- R5a: constant airflow ---
            thr_const = self._thr('R5a_airflow_const', 0.05, verbose)

            diff = v.diff().abs()
            daily_max_diff = diff.resample("1D").max().reindex(df.index, method="ffill")

            metric_const = (thr_const - daily_max_diff).clip(lower=0) / thr_const

            r5a, r5a_raw, r5a_mask, anomalies, anomaly_sources = self._apply_rule(
                r5a, r5a_raw, r5a_mask, metric_const, 0.5, col, anomalies, anomaly_sources
            )

            # Range
            daily_range = (v.resample("1D").max() - v.resample("1D").min()).abs()
            thr_rng = self._thr('R5b_airflow_range', 0.35, verbose)

            r5b, r5b_raw, r5b_mask, anomalies, anomaly_sources = \
                self._apply_rule(r5b, r5b_raw, r5b_mask, daily_range, thr_rng, col, anomalies, anomaly_sources)

        rule_scores['R5a_airflow_const'] = r5a
        rule_scores['R5b_airflow_range'] = r5b
        metrics['R5a_airflow_const'] = r5a_raw
        metrics['R5b_airflow_range'] = r5b_raw
        rule_masks['R5a_airflow_const'] = r5a_mask
        rule_masks['R5b_airflow_range'] = r5b_mask

        # ---------------- R6: Forced dates ----------------
        forced_mask = pd.Series(False, index=df.index)
        date_index = pd.to_datetime(df.index)

        for d in ["2025-05-28", "2026-02-13"]:
            day_mask = (date_index.date == pd.Timestamp(d).date())

            for col in df.columns:
                v = pd.to_numeric(df[col], errors='coerce')
                day = v[day_mask]
                if day.empty:
                    continue

                p01 = day.quantile(0.01)
                p99 = day.quantile(0.99)
                mask = ((v < p01) | (v > p99)) & day_mask

                forced_mask |= mask
                anomalies |= mask
                anomaly_sources[col] |= mask

        rule_scores['R6_iqr_dates'] = forced_mask.astype(float)
        metrics['R6_iqr_dates'] = forced_mask.astype(float)
        rule_masks['R6_iqr_dates'] = forced_mask

        return anomalies, rule_scores, metrics, rule_masks

    # =======================  LUCRETILLI  ========================
    def _apply_lucretili_rules(self, anomalies, anomaly_sources, df=None, verbose=True):
        if df is None:
            df = self.raw_df

        df = df.sort_index()
        rule_scores, metrics, rule_masks = {}, {}, {}

        w1h  = self._steps("1h")
        w4h  = self._steps("4h")
        w6h  = self._steps("6h")
        w24h = self._steps("24h")

        # R1 — Crackmeters (2 reglas)
        r1a, r1a_raw, r1a_mask = self._empty()
        r1b, r1b_raw, r1b_mask = self._empty()

        for col in [c for c in df.columns if "crack" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")

            # --- normalization range ---
            v_min = v.min()
            v_max = v.max()
            v_range = (v_max - v_min) + 1e-9

            # ---------- R1a: step change (normalized) ----------
            step_change = (
                v.rolling(w1h, min_periods=1).max()
                - v.rolling(w1h, min_periods=1).min()
            )
            step_norm = step_change / v_range

            thr_step = self._thr("R1a_crack_step", 0.01, verbose)   # 1% threshold

            r1a, r1a_raw, r1a_mask, anomalies, anomaly_sources = self._apply_rule(
                r1a, r1a_raw, r1a_mask, step_norm, thr_step, col, anomalies, anomaly_sources
            )

            # ---------- R1b: offset (normalized) ----------
            rolling_mean_24h = v.rolling(w24h, min_periods=1).mean()
            offset = (v - rolling_mean_24h).abs()
            offset_norm = offset / v_range

            thr_off = self._thr("R1b_crack_offset", 0.02, verbose)  # 2% threshold

            r1b, r1b_raw, r1b_mask, anomalies, anomaly_sources = self._apply_rule(
                r1b, r1b_raw, r1b_mask, offset_norm, thr_off, col, anomalies, anomaly_sources
            )

        rule_scores["R1a_crack_step"] = r1a
        rule_scores["R1b_crack_offset"] = r1b
        metrics["R1a_crack_step"] = r1a_raw
        metrics["R1b_crack_offset"] = r1b_raw
        rule_masks["R1a_crack_step"] = r1a_mask
        rule_masks["R1b_crack_offset"] = r1b_mask


        # R2 — Tiltmeters (3 reglas)
        r2a, r2a_raw, r2a_mask = self._empty()   # step ratio
        r2b, r2b_raw, r2b_mask = self._empty()   # drift
        r2c, r2c_raw, r2c_mask = self._empty()   # offset

        for col in [c for c in df.columns if "tilt" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")

            # --- normalization range ---
            v_min = v.min()
            v_max = v.max()
            v_range = (v_max - v_min) + 1e-9

            # ---------- R2a: step ratio (normalized) ----------
            roll_max = v.rolling(w1h, min_periods=1).max()
            roll_min = v.rolling(w1h, min_periods=1).min()
            step = roll_max - roll_min
            step_norm = step / v_range

            thr_step = self._thr("R2a_tilt_step", 0.01, verbose)  # 1%

            r2a, r2a_raw, r2a_mask, anomalies, anomaly_sources = self._apply_rule(
                r2a, r2a_raw, r2a_mask, step_norm, thr_step, col, anomalies, anomaly_sources
            )
            
            # ---------- R2c: offset from 24h mean (normalized) ----------
            rolling_mean_24h = v.rolling(w24h, min_periods=1).mean()
            offset = (v - rolling_mean_24h).abs()
            offset_norm = offset / v_range

            thr_off = self._thr("R2b_tilt_offset", 0.02, verbose)  # 2%

            r2c, r2c_raw, r2c_mask, anomalies, anomaly_sources = self._apply_rule(
                r2c, r2c_raw, r2c_mask, offset_norm, thr_off, col, anomalies, anomaly_sources
            )

        rule_scores["R2a_tilt_step"] = r2a
        rule_scores["R2b_tilt_offset"] = r2c
        metrics["R2a_tilt_step"] = r2a_raw
        metrics["R2b_tilt_offset"] = r2c_raw
        rule_masks["R2a_tilt_step"] = r2a_mask
        rule_masks["R2b_tilt_offset"] = r2c_mask


        # R3 — Accelerometers (1 regla)
        r3, r3_raw, r3_mask = self._empty()

        for col in [c for c in df.columns if "acc" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")
            std_val = v.std()
            if std_val == 0 or np.isnan(std_val):
                continue
            z = (v - v.mean()).abs() / std_val
            thr = self._thr("R3_vib_zscore", 5, verbose)

            r3, r3_raw, r3_mask, anomalies, anomaly_sources = self._apply_rule(
                r3, r3_raw, r3_mask, z, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R3_vib_zscore"] = r3
        metrics["R3_vib_zscore"] = r3_raw
        rule_masks["R3_vib_zscore"] = r3_mask

        # R4 — Humidity (3 reglas)
        r4a, r4a_raw, r4a_mask = self._empty()   # >90% 6h
        r4b, r4b_raw, r4b_mask = self._empty()   # >85% 7d
        r4c, r4c_raw, r4c_mask = self._empty()   # rapid change

        for col in [c for c in df.columns if "hum" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")

            # ---------- R4a: >90% for 6h ----------
            thr_a = self._thr("R4a_hum_spike_6h", 0.90, verbose)

            roll_6h = v.rolling(w6h, min_periods=w6h).mean() 
            metric_a = roll_6h

            r4a, r4a_raw, r4a_mask, anomalies, anomaly_sources = self._apply_rule(
                r4a, r4a_raw, r4a_mask, metric_a, thr_a, col, anomalies, anomaly_sources
            )

            # ---------- R4b: >85% for 7 days ----------
            thr_b = self._thr("R4b_hum_persist_7d", 0.85, verbose)

            daily_mean = v.resample("1D").mean() 
            roll_7d = daily_mean.rolling(7, min_periods=7).mean()
            metric_b = roll_7d.reindex(df.index, method="ffill")

            r4b, r4b_raw, r4b_mask, anomalies, anomaly_sources = self._apply_rule(
                r4b, r4b_raw, r4b_mask, metric_b, thr_b, col, anomalies, anomaly_sources
            )

            # ---------- R4c: rapid change >15% in 1h ----------
            thr_c = self._thr("R4c_hum_rapid_1h", 0.15, verbose)

            delta = v.diff().abs()
            delta_norm = delta 
            metric_c = delta_norm.rolling(w1h, min_periods=1).max()

            r4c, r4c_raw, r4c_mask, anomalies, anomaly_sources = self._apply_rule(
                r4c, r4c_raw, r4c_mask, metric_c, thr_c, col, anomalies, anomaly_sources
            )

        rule_scores["R4a_hum_spike_6h"] = r4a
        rule_scores["R4b_hum_persist_7d"] = r4b
        rule_scores["R4c_hum_rapid_1h"] = r4c
        metrics["R4a_hum_spike_6h"] = r4a_raw
        metrics["R4b_hum_persist_7d"] = r4b_raw
        metrics["R4c_hum_rapid_1h"] = r4c_raw
        rule_masks["R4a_hum_spike_6h"] = r4a_mask
        rule_masks["R4b_hum_persist_7d"] = r4b_mask
        rule_masks["R4c_hum_rapid_1h"] = r4c_mask

        # R5 — Visitors (1 regla)
        r5, r5_raw, r5_mask = self._empty()

        for col in [c for c in df.columns if "counter" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")
            thr = self._thr("R5_visitors_peak", 3, verbose)

            r5, r5_raw, r5_mask, anomalies, anomaly_sources = self._apply_rule(
                r5, r5_raw, r5_mask, v, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R5_visitors"] = r5
        metrics["R5_visitors"] = r5_raw
        rule_masks["R5_visitors"] = r5_mask

        # R6 — Rain (2 reglas)
        r6a, r6a_raw, r6a_mask = self._empty()   # 1h/24h mean
        r6b, r6b_raw, r6b_mask = self._empty()   # 7d cumulative

        rain_cols = [c for c in df.columns if "rain" in c.lower()]
        w24h_steps = self._steps("24h")

        for col in rain_cols:
            v = pd.to_numeric(df[col], errors="coerce")

            # ---------- R6a: 1h or 24h mean ----------
            thr_1h = self._thr("R6a_rain_1h", 5, verbose)
            thr_24h = self._thr("R6a_rain_24h", 20, verbose)

            r1h = v.rolling(w1h, min_periods=1).mean()
            r24h = v.rolling(w24h_steps, min_periods=1).mean()

            metric_a = self._max_aligned(self._align(r1h), self._align(r24h))
            thr_a = max(thr_1h, thr_24h)

            r6a, r6a_raw, r6a_mask, anomalies, anomaly_sources = self._apply_rule(
                r6a, r6a_raw, r6a_mask, metric_a, thr_a, col, anomalies, anomaly_sources
            )

            # ---------- R6b: 7-day cumulative ----------
            rain_daily = v.resample("1D").sum()
            cum7d = rain_daily.rolling(7, min_periods=1).sum().reindex(df.index, method="ffill")

            thr_7d = self._thr("R6b_rain_cum7d", 40, verbose)

            r6b, r6b_raw, r6b_mask, anomalies, anomaly_sources = self._apply_rule(
                r6b, r6b_raw, r6b_mask, cum7d, thr_7d, col, anomalies, anomaly_sources
            )

        rule_scores["R6a_rain"] = r6a
        rule_scores["R6b_rain_cum7d"] = r6b
        metrics["R6a_rain"] = r6a_raw
        metrics["R6b_rain_cum7d"] = r6b_raw
        rule_masks["R6a_rain"] = r6a_mask
        rule_masks["R6b_rain_cum7d"] = r6b_mask

        # R7 — Wind (1 regla)
        r7, r7_raw, r7_mask = self._empty()

        for col in [c for c in df.columns if "wind" in c.lower() and "dir" not in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")
            baseline = v.rolling(w24h, min_periods=1).mean()
            metric_peak = v / (baseline + 1e-9)

            thr_peak = self._thr("R7a_wind_peak_ratio", 5, verbose)

            r7, r7_raw, r7_mask, anomalies, anomaly_sources = self._apply_rule(
                r7, r7_raw, r7_mask, metric_peak, thr_peak, col, anomalies, anomaly_sources
            )
            

        rule_scores["R7_wind"] = r7
        metrics["R7_wind"] = r7_raw
        rule_masks["R7_wind"] = r7_mask


        # R8 — Barometer (1 regla)
        r8, r8_raw, r8_mask = self._empty()
        w6h_steps = self._steps("6h")

        for col in [c for c in df.columns if "baro" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")
            r6h = (
                v.rolling(w6h_steps, min_periods=1).max()
                - v.rolling(w6h_steps, min_periods=1).min()
            )
            thr = self._thr("R8_baro_range", 300, verbose)

            r8, r8_raw, r8_mask, anomalies, anomaly_sources = self._apply_rule(
                r8, r8_raw, r8_mask, r6h, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R8_baro"] = r8
        metrics["R8_baro"] = r8_raw
        rule_masks["R8_baro"] = r8_mask
        
        # ---------------- R9: Temperature ----------------
        r9a, r9a_raw, r9a_mask = self._empty()   # R9a: >40°C 4h
        r9b, r9b_raw, r9b_mask = self._empty()   # R9b: daily mean >=35°C 5d
        r9c, r9c_raw, r9c_mask = self._empty()   # R9c: 3‑day mean > threshold
        r9d, r9d_raw, r9d_mask = self._empty()   # R9d: rapid change >5°C in 1h

        for col in [c for c in df.columns if "temp" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")

            # ---------- R9a: >40°C for 4 consecutive hours ----------
            thr_a = self._thr("R9a_temp_high_st", 40, verbose)

            roll_4h = v.rolling(w4h, min_periods=w4h).mean()
            metric_a = roll_4h

            r9a, r9a_raw, r9a_mask, anomalies, anomaly_sources = self._apply_rule(
                r9a, r9a_raw, r9a_mask, metric_a, thr_a, col, anomalies, anomaly_sources
            )

            # ---------- R9b: daily mean >=35°C for 5 consecutive days ----------
            thr_b = self._thr("R9b_temp_high_lt", 35, verbose)

            daily_mean = v.resample("1D").mean()
            roll_5d = daily_mean.rolling(5, min_periods=5).mean()
            metric_b = roll_5d.reindex(df.index, method="ffill")

            r9b, r9b_raw, r9b_mask, anomalies, anomaly_sources = self._apply_rule(
                r9b, r9b_raw, r9b_mask, metric_b, thr_b, col, anomalies, anomaly_sources
            )

            # ---------- R9c: 3‑day mean > threshold ----------
            thr_c = self._thr("R9c_temp_drop_st", 32, verbose)

            mean_3d = daily_mean.rolling(3, min_periods=3).mean()
            metric_c = mean_3d.reindex(df.index, method="ffill")

            r9c, r9c_raw, r9c_mask, anomalies, anomaly_sources = self._apply_rule(
                r9c, r9c_raw, r9c_mask, metric_c, thr_c, col, anomalies, anomaly_sources
            )

            # ---------- R9d: rapid change >5°C in 1h ----------
            thr_d = self._thr("R9d_temp_drop_lt", 5, verbose)

            delta_1h = v.diff().abs().rolling(w1h, min_periods=1).max()
            metric_d = delta_1h

            r9d, r9d_raw, r9d_mask, anomalies, anomaly_sources = self._apply_rule(
                r9d, r9d_raw, r9d_mask, metric_d, thr_d, col, anomalies, anomaly_sources
            )

        rule_scores["R9a_temp_high_st"] = r9a
        rule_scores["R9b_temp_high_lt"] = r9b
        rule_scores["R9c_temp_drop_st"] = r9c
        rule_scores["R9d_temp_drop_lt"] = r9d

        metrics["R9a_temp_high_st"] = r9a_raw
        metrics["R9b_temp_high_lt"] = r9b_raw
        metrics["R9c_temp_drop_st"] = r9c_raw
        metrics["R9d_temp_drop_lt"] = r9d_raw

        rule_masks["R9a_temp_high_st"] = r9a_mask
        rule_masks["R9b_temp_high_lt"] = r9b_mask
        rule_masks["R9c_temp_drop_st"] = r9c_mask
        rule_masks["R9d_temp_drop_lt"] = r9d_mask

        # R10 — solar z score 
        r10, r10_raw, r10_mask = self._empty()

        for col in [c for c in df.columns if "gnss" in c.lower()]:
            v = pd.to_numeric(df[col], errors="coerce")

            # --- daily mean and std ---
            daily_mean = v.resample("1D").mean()
            daily_std = v.resample("1D").std()

            # avoid division by zero
            daily_std = daily_std.replace(0, np.nan)

            # --- z-score ---
            z = ((daily_mean - daily_mean.mean()) / daily_std).abs()

            # threshold
            thr = self._thr("R10_solar_zscore", 1.0, verbose)   # typical z-score threshold

            # feed into rule engine
            r10, r10_raw, r10_mask, anomalies, anomaly_sources = self._apply_rule(
                r10, r10_raw, r10_mask, z, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R10_solar_zscore"] = r10
        metrics["R10_solar_zscore"] = r10_raw
        rule_masks["R10_solar_zscore"] = r10_mask

        return anomalies, rule_scores, metrics, rule_masks

    # =======================  RANVERSO  ==========================
    def _apply_ranverso_rules(self, anomalies, anomaly_sources, df=None, verbose=True):
        if df is None:
            df = self.raw_df

        df = df.sort_index()
        rule_scores, metrics, rule_masks = {}, {}, {}

        # ---------------- R1: Indoor humidity > 80% ----------------
        r1, r1_raw, r1_mask = self._empty()

        for col in [c for c in df.columns if "hum" in c and "in" in c]:
            v = pd.to_numeric(df[col], errors="coerce")
            thr = self._thr("R1_hum_inside", 80, verbose)

            r1, r1_raw, r1_mask, anomalies, anomaly_sources = self._apply_rule(
                r1, r1_raw, r1_mask, v, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R1_hum_inside"] = r1
        metrics["R1_hum_inside"] = r1_raw
        rule_masks["R1_hum_inside"] = r1_mask

        # ---------------- R2: Rain every day for 7 days ----------------
        r2, r2_raw, r2_mask = self._empty()
        rain_cols = [c for c in df.columns if "rain" in c.lower() and "acc" not in c.lower()]
        w7d = self._steps("7D")

        for col in rain_cols:
            v = pd.to_numeric(df[col], errors="coerce")
            #rain_7d_diff = v - v.shift(w7d)

            # threshold
            thr = self._thr("R2_rain_7d_diff", 5, verbose)
            rain_7d_sum = v.rolling(w7d, min_periods=w7d).sum()

            # feed into rule engine using the metric
            r2, r2_raw, r2_mask, anomalies, anomaly_sources = self._apply_rule(
                r2, r2_raw, r2_mask, rain_7d_sum, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R2_rain_7d"] = r2
        metrics["R2_rain_7d"] = r2_raw
        rule_masks["R2_rain_7d"] = r2_mask

        # ---------------- R3: No difference soil vs weather temp ----------------
        r3, r3_raw, r3_mask = self._empty()

        temp1_cols = [c for c in df.columns if "soilmoisture_temp" in c.lower()]
        temp2_cols = [c for c in df.columns if "weather" in c.lower() and "temp" in c.lower()]

        if temp1_cols and temp2_cols:
            t1 = pd.to_numeric(df[temp1_cols[0]], errors="coerce")
            t2 = pd.to_numeric(df[temp2_cols[0]], errors="coerce")

            diff = (t1 - t2).abs()
            thr = self._thr("R3_temp_nodiff", 1, verbose)

            metric = (thr - diff).clip(lower=0) / thr

            r3, r3_raw, r3_mask, anomalies, anomaly_sources = self._apply_rule(
                r3, r3_raw, r3_mask, metric, 0.5, "temp_nodiff", anomalies, anomaly_sources
            )

        rule_scores["R3_temp_nodiff"] = r3
        metrics["R3_temp_nodiff"] = r3_raw
        rule_masks["R3_temp_nodiff"] = r3_mask
        
        # ---------------- R4: Soil moisture range after 2025-11-15 ----------------
        r4_hi, r4_hi_raw, r4_hi_mask = self._empty()
        r4_lo, r4_lo_raw, r4_lo_mask = self._empty()

        start_date = pd.Timestamp("2025-11-15")
        soil_cols = [c for c in df.columns
            if "moisture" in c.lower() and "temp" not in c.lower() and "hum" not in c.lower()
        ]

        for col in soil_cols:
            v = pd.to_numeric(df[col], errors="coerce")
            mask_period = df.index >= start_date

            # High
            thr_hi = self._thr("R4a_soil_high", 2.5, verbose)
            metric_hi = v.where(mask_period, 0)
            r4_hi, r4_hi_raw, r4_hi_mask, anomalies, anomaly_sources = self._apply_rule(
                r4_hi, r4_hi_raw, r4_hi_mask, metric_hi, thr_hi, col, anomalies, anomaly_sources
            )

            # Low
            thr_lo = self._thr("R4b_soil_low", 0.5, verbose)
            metric_lo = -v.where(mask_period, 0)
            r4_lo, r4_lo_raw, r4_lo_mask, anomalies, anomaly_sources = self._apply_rule(
                r4_lo, r4_lo_raw, r4_lo_mask, metric_lo, -thr_lo, col, anomalies, anomaly_sources
            )

        rule_scores["R4a_soil_high"] = r4_hi
        rule_scores["R4b_soil_low"] = r4_lo
        metrics["R4a_soil_high"] = r4_hi_raw
        metrics["R4b_soil_low"] = r4_lo_raw
        rule_masks["R4a_soil_high"] = r4_hi_mask
        rule_masks["R4b_soil_low"] = r4_lo_mask
        '''
        # ---------------- R5: Airflow out of range ----------------
        r5_hi, r5_hi_raw, r5_hi_mask = self._empty()
        r5_lo, r5_lo_raw, r5_lo_mask = self._empty()

        airflow_cols = [c for c in df.columns if "airflow" in c.lower()]

        for col in airflow_cols:
            v = pd.to_numeric(df[col], errors="coerce")

            thr_hi = self._thr("R5a_airflow_high", 1.5, verbose)
            thr_lo = self._thr("R5b_airflow_low", 0.5, verbose)

            # High
            r5_hi, r5_hi_raw, r5_hi_mask, anomalies, anomaly_sources = self._apply_rule(
                r5_hi, r5_hi_raw, r5_hi_mask, v, thr_hi, col, anomalies, anomaly_sources
            )

            # Low
            r5_lo, r5_lo_raw, r5_lo_mask, anomalies, anomaly_sources = self._apply_rule(
                r5_lo, r5_lo_raw, r5_lo_mask, -v, -thr_lo, col, anomalies, anomaly_sources
            )

        rule_scores["R5a_airflow_high"] = r5_hi
        rule_scores["R5b_airflow_low"] = r5_lo
        metrics["R5a_airflow_high"] = r5_hi_raw
        metrics["R5b_airflow_low"] = r5_lo_raw
        rule_masks["R5a_airflow_high"] = r5_hi_mask
        rule_masks["R5b_airflow_low"] = r5_lo_mask
        '''
        return anomalies, rule_scores, metrics, rule_masks

    # =====================  SCHENKENBERG  ======================
    def _apply_schenkenberg_rules(self, anomalies, anomaly_sources, df=None, verbose=True):
        if df is None:
            df = self.raw_df

        df = df.sort_index()
        rule_scores, metrics, rule_masks = {}, {}, {}

        w1h = self._steps("1h")
        w24h = self._steps("24h")

        # ---------------- R1: Low temp + high humidity ----------------
        r1, r1_raw, r1_mask = self._empty() # no apply rule bc 2 together

        temp_cols = [c for c in df.columns if "temp" in c.lower()]
        hum_cols = [c for c in df.columns if "hum" in c.lower()]

        for tcol in temp_cols:
            t_vals = pd.to_numeric(df[tcol], errors="coerce")
            for hcol in hum_cols:
                h_vals = pd.to_numeric(df[hcol], errors="coerce")

                thr_t = self._thr("R1_temp_low", 5, verbose)
                thr_h = self._thr("R1_hum_high", 90, verbose)

                score_t = self._align(self._sigmoid_score(-t_vals, -thr_t))
                score_h = self._align(self._sigmoid_score(h_vals, thr_h))
                score = pd.concat([score_t, score_h], axis=1).min(axis=1)

                mask = ((t_vals < thr_t) & (h_vals > thr_h)).reindex(df.index, fill_value=False)

                r1 = self._max_aligned(r1, score)
                r1_raw = self._max_aligned(r1_raw, self._align(h_vals))  # arbitrary metric
                r1_mask |= mask

                anomalies |= mask
                anomaly_sources[tcol] |= mask
                anomaly_sources[hcol] |= mask

        rule_scores["R1_temp_hum"] = r1
        metrics["R1_temp_hum"] = r1_raw
        rule_masks["R1_temp_hum"] = r1_mask

        # ---------------- R2: Crackmeter step change ----------------
        r2, r2_raw, r2_mask = self._empty()

        for col in [c for c in df.columns if "crack" in c.lower()]:
            vals = pd.to_numeric(df[col], errors="coerce")

            # --- normalization range ---
            v_min = vals.min()
            v_max = vals.max()
            v_range = (v_max - v_min) + 1e-9

            # --- step metric ---
            roll_mean = vals.rolling(w1h, min_periods=1).mean()
            step = (vals - roll_mean).abs()

            # --- normalized step (percentage of dynamic range) ---
            norm_step = step / v_range

            # threshold in % of range
            thr = self._thr("R2_crack_step", 0.005, verbose)   # 0.5%

            # feed into rule engine
            r2, r2_raw, r2_mask, anomalies, anomaly_sources = self._apply_rule(
                r2, r2_raw, r2_mask, norm_step, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R2_crack_step"] = r2
        metrics["R2_crack_step"] = r2_raw
        rule_masks["R2_crack_step"] = r2_mask

        # ---------------- R3: Vibration z-score ----------------
        r3, r3_raw, r3_mask = self._empty()

        for col in [c for c in df.columns if "pga" in c.lower() or "rms" in c.lower()]:
            vals = pd.to_numeric(df[col], errors="coerce")
            std = vals.std()
            if std == 0 or np.isnan(std):
                continue
            z = (vals - vals.mean()).abs() / std

            thr = self._thr("R3_vib_zscore", 3, verbose)
            r3, r3_raw, r3_mask, anomalies, anomaly_sources = self._apply_rule(
                r3, r3_raw, r3_mask, z, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R3_vib_zscore"] = r3
        metrics["R3_vib_zscore"] = r3_raw
        rule_masks["R3_vib_zscore"] = r3_mask


        r4, r4_raw, r4_mask = self._empty()
        for col in [c for c in df.columns if "pga" in c.lower()]:
            vals = pd.to_numeric(df[col], errors="coerce")
            std = vals.std()
            if std == 0 or np.isnan(std):
                continue
            z = (vals - vals.mean()).abs() / std

            thr = self._thr("R4_pga", 0.5, verbose)
            r4, r4_raw, r4_mask, anomalies, anomaly_sources = self._apply_rule(
                r4, r4_raw, r4_mask, z, thr, col, anomalies, anomaly_sources
            )

        rule_scores["R4_pga"] = r4
        metrics["R4_pga"] = r4_raw
        rule_masks["R4_pga"] = r4_mask

        # ---------------- R5: Soil moisture instability ----------------
        '''
        r5a, r5a_raw, r5a_mask = self._empty()
        r5b, r5b_raw, r5b_mask = self._empty()

        soil_cols = [c for c in df.columns if "moisture" in c.lower() and "temp" not in c.lower() and "ec" not in c.lower()]

        for col in soil_cols:
            vals = pd.to_numeric(df[col], errors="coerce")

            thr_low = self._thr("R5_soil_high", 1.15, verbose)
            thr_rng = self._thr("R5_soil_range", 0.15, verbose)

            # Low
            r5a, r5a_raw, r5a_mask, anomalies, anomaly_sources = self._apply_rule(
                r5a, r5a_raw, r5a_mask, vals, thr_low, col, anomalies, anomaly_sources
            )

            # Range
            daily_range = (vals.resample("1D").max() - vals.resample("1D").min()).abs()
            r5b, r5b_raw, r5b_mask, anomalies, anomaly_sources = self._apply_rule(
                r5b, r5b_raw, r5b_mask, daily_range, thr_rng, col, anomalies, anomaly_sources
            )

        rule_scores["R5a_soil_high"] = r5a
        rule_scores["R5b_soil_range"] = r5b
        metrics["R5a_soil_high"] = r5a_raw
        metrics["R5b_soil_range"] = r5b_raw
        rule_masks["R5a_soil_high"] = r5a_mask
        rule_masks["R5b_soil_range"] = r5b_mask
        '''
        return anomalies, rule_scores, metrics, rule_masks