import numpy as np
import os, torch, datetime, json, csv
from config.reporter import Reporter
import utils.helpers_plots as hp

def abAD_index(scores, threshold):
    '''
    Computes the proposed abAD index to assess anomaly separation in an unsupervised setting.
    '''
    
    anomalies_test = scores > threshold
    normal_instances = scores[~anomalies_test]
    anomaly_instances = scores[anomalies_test]
    
    n_samples = len(scores)
    n_normal = len(normal_instances)
    n_anomaly = len(anomaly_instances)

    mean_normal = np.mean(normal_instances) #centroid_normal
    if n_normal == 0 or n_anomaly == 0:  
        return 0  

    #between cluster scatter to maximize distance of different clusters
    #B_k = (n_normal * (mean_normal - mean_all) ** 2) + (n_anomaly * (mean_anomaly - mean_all) ** 2)
    B_k = 0
    for normal in normal_instances:
        for anomaly in anomaly_instances:
            squared_distance = (normal - anomaly) ** 2
            B_k += squared_distance
    
    #within cluster just in the normal data points; anomalies do not need to be so dense
    W_k = np.sum((normal_instances - mean_normal) ** 2) #wss just for normal data
    if W_k == 0:
        return np.inf  #avoid div 0
    
    max_possible = (np.max(anomaly_instances) - np.min(normal_instances)) ** 2
    abad_score = (B_k / W_k) / max_possible
    
    return abad_score

def fit_mahalanobis(latent_train):
    "Estimates the mean vector and inverse covariance matrix for Mahalanobis distance computation."
    B, T, F = latent_train.shape
    z = latent_train.reshape(B * T, F)
    mu = z.mean(dim=0)
    cov = torch.cov(z.T)
    cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0]))
    return mu, cov_inv

def mahalanobis_distance(latent, mu, cov_inv):
    " Computes the Mahalanobis distance between latent representations and the training distribution."
    diff = latent - mu
    m_dist = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
    return m_dist

def get_scores(dataloader, model, device, beta, rec_refs, fc_refs,
               alpha_ema=0.5, isprint=False, block_attn=False, block_tcn=False,
               ctx=None, revin=False, anom_indices=None,
               plot_diagnostics=False, feature_names=None,
               anomaly_ranges=None, diagnostics_save_path=None,
               rho=0.05):
    "Computes reconstruction, forecasting, and combined anomaly scores for a dataset."
    model.eval()

    scores = []
    feat_contribution = []
    rec_scores = []
    fc_scores = []

    rec_raw_all = []     # normalized feature-wise rec errors [B,T,F]
    fc_raw_all = []      # normalized feature-wise fc errors [B,F]
    rec_err_raw_all = [] # RAW (pre-normalization) rec errors [B,T,F]
    fc_err_raw_all = []  # RAW (pre-normalization) fc errors [B,F]

    all_y_rec_pred, all_y_fc_pred = [], []
    all_y_rec, all_y_fc = [], []

    with torch.no_grad():
        for idx, (x, y_rec, y_fc, alive, _) in enumerate(dataloader):

            x = x.to(device)
            y_rec = y_rec.to(device)
            y_fc = y_fc.to(device)
            alive = alive.to(device)
            alive = alive[:, :-1, :]

            y_pred1, y_pred2 = model(x, block_attn=block_attn, block_tcn=block_tcn, revin=revin)

            # ---------------------ERRORS---------------------
            rec_err = torch.abs(y_pred1 - y_rec)   # [B,T,F]
            fc_err  = torch.abs(y_pred2 - y_fc)    # [B,F]

            rec_err_raw_all.append(rec_err.cpu())
            fc_err_raw_all.append(fc_err.cpu())

            rec_err_np = rec_err.cpu().numpy()   # [B,T,F]
            fc_err_np  = fc_err.cpu().numpy()    # [B,F]
            # Normalize along dim 2 (features)
            rec_n_np = rec_err_np / rec_refs.reshape(1, 1, -1)
            fc_n_np  = fc_err_np  / fc_refs.reshape(1, -1)
            rec_n = torch.from_numpy(rec_n_np).to(device)
            fc_n  = torch.from_numpy(fc_n_np).to(device)

            rec_raw_all.append(rec_n.cpu())
            fc_raw_all.append(fc_n.cpu())

            # -------------------SCORES---------------------
            ema_feat = torch.zeros_like(rec_n)
            ema_feat[:, 0, :] = rec_n[:, 0, :]
            for t in range(1, rec_n.shape[1]):
                ema_feat[:, t, :] = (alpha_ema * ema_feat[:, t - 1, :]
                                      + (1 - alpha_ema) * rec_n[:, t, :])

            rec_feat_score = ema_feat[:, -1]
            fc_feat_score = fc_n

            rec_score = rec_feat_score.mean(-1)
            fc_score  = fc_feat_score.mean(-1)
            score = beta * rec_score + (1 - beta) * fc_score
            feat_score = beta * rec_feat_score + (1 - beta) * fc_feat_score

            scores.append(score.cpu().numpy())
            feat_contribution.append(feat_score.cpu().numpy())
            rec_scores.append(rec_score.cpu().numpy())
            fc_scores.append(fc_score.cpu().numpy())

            all_y_rec_pred.append(y_pred1.cpu())
            all_y_fc_pred.append(y_pred2.cpu())
            all_y_rec.append(y_rec.cpu())
            all_y_fc.append(y_fc.cpu())

    all_scores = np.concatenate(scores, axis=0)
    all_feats = np.concatenate(feat_contribution, axis=0)
    rec_scores = np.concatenate(rec_scores, axis=0)
    fc_scores = np.concatenate(fc_scores, axis=0)

    if isprint:
        def print_scores_for_indices(name, arr, indices):
            print(f"\n--- {name} ---")
            idxs = sorted(indices, key=lambda i: arr[i], reverse=True)
            for i in idxs:
                print(f"Window {i:4d} | score={arr[i]:.4f}")

        print("\n=== SCORES FOR INJECTED ANOMALIES ===")
        print_scores_for_indices("RECON SCORE (EMA)", rec_scores, anom_indices)
        print_scores_for_indices("FORECAST SCORE", fc_scores, anom_indices)
        print_scores_for_indices("FINAL FUSED SCORE", all_scores, anom_indices)
        print("\n")

        for w in anom_indices:
            hp.plot_instance(y_rec_pred=all_y_rec_pred[w], y_fc_pred=all_y_fc_pred[w],
                              y_rec=all_y_rec[w], y_fc=all_y_fc[w], idx=w, ctx=ctx,
                              feat_contrib=all_feats)

    if plot_diagnostics:
        hp.plot_feature_error_diagnostics(
            rec_err_raw_all, fc_err_raw_all, rec_raw_all, fc_raw_all,
            feature_names=feature_names, anomaly_ranges=anomaly_ranges,
            rho=rho, save_path=diagnostics_save_path
        )

    return all_scores, all_feats, rec_scores, fc_scores, rec_raw_all, fc_raw_all

def get_all_scores(model, ctx, train_loader, val_loader=None, test_loader=None,
                    beta=0.5, device="cuda", return_test=False, real_anomalies=None,
                    rho=0.05, scale=None):
    "Computes anomaly scores for the train, validation, or test sets."

    print("Computing latents and scores...")
    experiment_path = ctx["experiment_path"]
    block_attn, block_tcn = False, False
    if "noattn" in experiment_path:
        block_attn = True
        print("\t with blocked feature path!!!!")
    elif "notcn" in experiment_path:
        block_tcn = True
        print("\t with blocked temporal path!!!!")
    else:
        print("\t with both paths")

    if "norevin" in ctx["experiment_path"]:
        revin = False
        print("\t deactivated revin")
    else:
        revin = True
        print("\t activated revin")

    # Reference for error normalization: pooled train+val, fit ONCE.
    rec_refs, fc_refs = scale, scale

    if not return_test:
        scores_train, feats_train, rec_train, fc_train, _, _ = get_scores(
            train_loader, model, device, beta, rec_refs, fc_refs,
            block_attn=block_attn, block_tcn=block_tcn, revin=revin, plot_diagnostics=True, feature_names=ctx["feature_names"], diagnostics_save_path=os.path.join(ctx["experiment_path"], "feature_error_diagnostics_train.png"))
        scores_val, feats_val, rec_val, fc_val, _, _ = get_scores(
            val_loader, model, device, beta, rec_refs, fc_refs,
            block_attn=block_attn, block_tcn=block_tcn, revin=revin)

        print(f"Mean and std scores in Train ({scores_train.mean():.4f}, {scores_train.std():.4f}), "
              f"Val ({scores_val.mean():.4f}, {scores_val.std():.4f})")

        return scores_train, scores_val

    else:
        scores_test, feats_test, rec_test, fc_test, _, _ = get_scores(
            test_loader, model, device, beta, rec_refs, fc_refs,
            isprint=False, anom_indices=real_anomalies, ctx=ctx,
            block_attn=block_attn, block_tcn=block_tcn, revin=revin)

        print(f"Mean and std scores in Test ({scores_test.mean():.4f}, {scores_test.std():.4f})")

        return scores_test, feats_test, rec_test, fc_test
        
def get_info_anomalies(feat_scores, scores_list, threshold, feature_names, tau, top_k=5, exclude_mask=None):
    '''
     Identifies detected anomalies and their most influential features.
    '''
    results = []
    # Índices donde el score global supera el umbral
    anom_indices = np.where(scores_list > threshold)[0]

    if exclude_mask is not None: # for real candidates
        anom_indices = anom_indices[~exclude_mask[anom_indices]]

    anom_indices = anom_indices.tolist()
    
    sorted_anoms = sorted(anom_indices,key=lambda i: scores_list[i],reverse=True)
    #print("\n===ANOMALIES DETECTED====")
    #print(anom_indices)
    print("\n=== ANOMALIES DETECTED (SORTED BY SCORE) ===")
    print(sorted_anoms)
    
    K = max(1, int(0.10 * len(scores_list)))
    top10_idx = np.argsort(scores_list)[::-1][:K]

    print(f"\n=== TOP {K} SCORES (10%) ===")
    print(top10_idx.tolist())


    
    for idx in anom_indices:
        feat = torch.tensor(feat_scores[idx], dtype=torch.float32)

        # normalize contributions
        contrib = feat / (feat.sum() + 1e-8)

        # top features
        top_vals, top_idxs = torch.topk(contrib, top_k)
        top_features = [feature_names[i] for i in top_idxs.tolist()]
        top_feature_idxs = top_idxs.tolist()
        #print(f"idx={idx} | top_feature_idxs={top_feature_idxs}")

        # confidence before tail
        conf = 1 / (1 + np.exp(-(scores_list[idx] - threshold)))
        
        #delta = max(scores_list[idx]-threshold,0)
        #conf = 0.5 + 0.5*(1-np.exp(-delta/tau))
        
        feature_contrib = {
            feat: round(float(val), 4)
            for feat, val in zip(top_features, top_vals)
        }
        
        results.append({
            "anomaly_index": int(idx),
            "score": round(float(scores_list[idx]), 4),
            "confidence_score": round(float(conf), 4),
            "top_feature_contributions": feature_contrib,
        })

    return results

def threshold_computation(scores_train_val, threshold_method="POT", percentile=95, quantile=0.050): 
    "Computes the anomaly detection threshold using the selected thresholding method"
    if threshold_method == "POT":
        t = np.percentile(scores_train_val, percentile)
        prelim_anom =  scores_train_val[scores_train_val > t] - t        
        if len(prelim_anom) == 0:
            raise ValueError("No samples exceed preliminary threshold percentile.")
        gamma, mu, sigma = genpareto.fit(prelim_anom)
        threshold = t + (sigma / gamma) * ((quantile * len(scores_train_val) / len(prelim_anom)) ** (-gamma) - 1)
        
    elif threshold_method == "stability-POT":
        print("Using stability-POT (unsupervised EVT)")

        # 1. Sweep percentiles (p)
        p_range = list(range(80, 98))  # tail-focused but broad
        gammas = []
        sigmas = []
        for p in p_range:
            t_candidate = np.percentile(scores_train_val, p)
            excess = scores_train_val[scores_train_val > t_candidate] - t_candidate
            excess = excess[excess > 1e-8]
            # require enough samples for EVT
            if len(excess) < 30:
                gammas.append(np.nan)
                sigmas.append(np.nan)
                continue
            try:
                gamma, loc, sigma = genpareto.fit(excess, floc=0)
                gammas.append(gamma)
                sigmas.append(sigma)
            except:
                gammas.append(np.nan)
                sigmas.append(np.nan)

        gammas = np.array(gammas)
        sigmas = np.array(sigmas)

        # 2. Smooth parameters (optional but recommended)
        def smooth(x, w=3):
            return np.convolve(x, np.ones(w)/w, mode='same')

        gammas_s = smooth(gammas)
        sigmas_s = smooth(sigmas)

        # 3. Compute stability score
        window = 3
        stability_scores = []
        for i in range(len(p_range) - window):
            g_window = gammas_s[i:i+window]
            s_window = sigmas_s[i:i+window]
            if np.any(np.isnan(g_window)) or np.any(np.isnan(s_window)):
                stability_scores.append(np.inf)
                continue
            g_std = np.std(g_window)
            s_std = np.std(s_window)
            stability_scores.append(g_std + s_std)
        stability_scores = np.array(stability_scores)

        # 4. Select best p (start of stable region)
        if len(stability_scores) == 0 or np.all(np.isinf(stability_scores)):
            print("Stability failed, fallback to percentile 95")
            best_p = 95
        else:
            best_idx = np.argmin(stability_scores)
            best_p = p_range[best_idx]

        print(f"Selected stable percentile p = {best_p}")
        
        # 5. Fit final GPD at chosen p
        t = np.percentile(scores_train_val, best_p)
        excess = scores_train_val[scores_train_val > t] - t
        excess = excess[excess > 1e-8]

        if len(excess) < 10:
            print("Too few excesses, fallback to percentile 95")
            threshold = np.percentile(scores_train_val, 95)
        else:
            gamma, loc, sigma = genpareto.fit(excess, floc=0)
            if abs(gamma) < 1e-6:
                threshold = t + sigma * np.log(len(scores_train_val) / len(excess))
            else:
                threshold = t + (sigma / gamma) * (
                    (quantile * len(scores_train_val) / len(excess)) ** (-gamma) - 1
                )
                
                t1= t + (sigma / gamma) * (
                    (0.001 * len(scores_train_val) / len(excess)) ** (-gamma) - 1
                )
                t2= t + (sigma / gamma) * (
                    (0.1 * len(scores_train_val) / len(excess)) ** (-gamma) - 1
                )
                print(f"with 0.001 and 0.1 {t1:.4f}, {t2:.4f}")
                
        print(sigma, gamma)

        print(f"Final threshold (stability-POT): {threshold:.4f}")
    
    elif threshold_method == "median":
        print("\n Using fixed threshold")
        # --- basic stats --- 
        mean_val = np.mean(scores_train_val) 
        median_val = np.median(scores_train_val) 
        std_val = np.std(scores_train_val) 
        # --- MAD --- 
        abs_dev = np.abs(scores_train_val - median_val) 
        mad_val = np.median(abs_dev) 
        # scaled MAD (std-equivalent) 
        mad_scaled = 1.4826 * mad_val
        # --- thresholds for comparison --- 
        thr_std = mean_val + 3 * std_val 
        thr_mad = median_val + 3 * mad_scaled 
        # --- prints --- 
        print(f"Mean: {mean_val:.6f}") 
        print(f"Median: {median_val:.6f}") 
        print(f"Std: {std_val:.6f}") 
        print(f"MAD: {mad_val:.6f}") 
        print(f"Scaled MAD (≈std): {mad_scaled:.6f}")
        print(f"Threshold (mean + 3*std): {thr_std:.6f}") 
        print(f"Threshold (median + 3*scaled MAD): {thr_mad:.6f}")
        threshold=thr_mad
        
    else:
        raise ValueError("Unidentified threshold selection method.")
    
    return threshold

def average_comparison():
    "Aggregates average experiment reports into a CSV comparison table"
    base_path = "models/checkpoints"

    # Find folders that end with '_average'
    folders = [f for f in os.listdir(base_path) if f.endswith("_average")]

    # Store all model names and their metrics
    model_metrics = []
    all_keys = set()

    for folder in folders:
        report_path = os.path.join(base_path, folder, "average_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                metrics = json.load(f)
                model_name = "_".join(folder.split("_")[:2])  # Keep only first 2 parts
                model_metrics.append((model_name, metrics))
                all_keys.update(metrics.keys())
        else:
            print(f"Warning: {report_path} not found.")

    # Sort the metric keys for consistent column order
    sorted_keys = sorted(all_keys)
    header = ["model"] + sorted_keys

    # Output file name with today's date
    today_str = datetime.today().strftime("%d%m%Y")
    output_file = os.path.join(base_path, f"comparison_{today_str}.csv")

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for model_name, metrics in model_metrics:
            row = [model_name]
            for key in sorted_keys:
                value = metrics.get(key, "")
                if isinstance(value, (float, int)):
                    value = round(value, 4)  # Changed to 4 decimal places
                row.append(value)
            writer.writerow(row)

    print(f" CSV saved to: {output_file}")

def average_public_report(args, data_bundle, task="pretext"):
    """
    Computes and stores the average evaluation metrics across all entities of a public dataset.
    """
    summary_metrics = data_bundle["summary_metrics"]
    
    if task == "pretext":
        keys_to_average = [
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_index",
            "abAD_score",
            "anomaly_num",
            "anomaly_percentage",
        ]
    elif task == "downstream":
        keys_to_average = [
            "accuracy",
            "precision",
            "recall",
            "F1 score",
            "ROC-AUC",
            "PR-AUC",
            "anomaly_num",
            "anomaly_percentage",
        ]
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'pretext' or 'downstream'.")

    # Compute mean across entities
    mean_test_metrics = {key: np.mean([m[key] for m in summary_metrics]) for key in keys_to_average}
    mean_test_metrics["total_training_time_s"] = data_bundle.get("total_training_time", 0.0)

    # Save to JSON using Reporter
    av_path = os.path.join(args.checkpoint, f"{args.arc}_{args.model_id}_{args.dataset}_average")
    os.makedirs(av_path, exist_ok=True)
    average_reporter = Reporter(av_path, "average_report.json")
    average_reporter.report("average_test_metrics", mean_test_metrics)

    # Print results
    print(f"\n=== Average public dataset metrics ({task}) ===")
    for k, v in mean_test_metrics.items():
        print(f"{k}: {v:.4f}")