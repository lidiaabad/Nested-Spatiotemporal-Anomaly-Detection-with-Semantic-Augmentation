import numpy as np
import pandas as pd
import math, os, torch
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import matplotlib.dates as mdates


def feature_plot(df_with_alive, feature_names, figure_path, n_features=None):
    "Plots all input features over time, highlighting inactive periods."

    data = df_with_alive.values
    x = pd.to_datetime(df_with_alive.index)
    
    for col in df_with_alive.columns:
        df_with_alive[col] = pd.to_numeric(df_with_alive[col],errors="coerce")
    data = df_with_alive.to_numpy(dtype=np.float64, na_value=np.nan)
    if n_features is None:
        n_features = len(feature_names) - 6

    fig, axes = plt.subplots(math.ceil(n_features / 4),4,figsize=(30, math.ceil(n_features / 4) * 3),sharex=True)
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        feature_col = data[:, i]
        alive_col = df_with_alive[f"{feature_names[i]}_alive"].values.astype(bool)
        diff = np.diff(alive_col.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            ax.axvspan(x[s],x[e - 1],color="lightgray",alpha=0.3,zorder=0,)

        ax.plot(x, feature_col, color="black", linewidth=1, alpha=0.8)

        ax.set_title(f"F{i}: {feature_names[i]}")
        ax.set_ylabel("Value")

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

def feature_input_model_plot(train_data,val_data, test_data, feature_names, figure_path,ek_rules=None,
    ek_rules_ds=None,events_ds=None,injected=None,):
    "Visualizes train, validation, and test feature values together with alive masks, events, EK rules, and injected anomalies."

    total_cols = train_data.shape[1]
    n_features = (total_cols - 6) // 2
    alive_block_start = train_data.shape[1] - n_features
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, n_rows * 3))
    axes = axes.flatten()

    data_all = np.concatenate([train_data, val_data, test_data], axis=0)

    x_all = np.arange(len(data_all))

    n_train = len(train_data)
    n_val = len(val_data)
    split_1 = n_train
    split_2 = n_train + n_val

    shifted_injected = None
    if injected is not None:
        shifted_injected = np.zeros(len(data_all))
        shifted_injected[split_2: split_2 + len(injected)] = injected

    for i in range(n_features):

        ax = axes[i]

        feature_col = data_all[:, i]
        alive_col = data_all[:, alive_block_start + i].astype(bool)


        diff = np.diff(alive_col.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            ax.axvspan(x_all[s],x_all[e - 1],color="lightgray",alpha=0.3,zorder=0, )

        ax.plot(x_all[:split_1],feature_col[:split_1],color="blue",alpha=0.7,linewidth=1,label="Train" if i == 0 else None, )
        ax.plot( x_all[split_1:split_2], feature_col[split_1:split_2], color="green", alpha=0.7, linewidth=1, label="Val" if i == 0 else None)
        ax.plot(x_all[split_2:],feature_col[split_2:],color="orange",alpha=0.7,linewidth=1,label="Test" if i == 0 else None,)

        if ek_rules is not None:
            idx = np.where(np.asarray(ek_rules) > 0.5)[0]
            ax.scatter(idx, feature_col[idx], color="red", marker="x", s=20, alpha=0.8,
                       label="EK" if i == 0 else None)

        if events_ds is not None:
            idx = np.where(np.asarray(events_ds) > 0.5)[0]
            ax.scatter(idx, feature_col[idx], color="black", marker="o", s=12, alpha=0.8,
                       label="Events" if i == 0 else None)

        if ek_rules_ds is not None:
            idx = np.where(np.asarray(ek_rules_ds) > 0.5)[0]
            ax.scatter(idx, feature_col[idx], color="purple", marker="+", s=30, alpha=0.8,
                       label="DS EK" if i == 0 else None)

        if shifted_injected is not None:
            idx = np.where(shifted_injected > 0.5)[0]
            ax.scatter(idx, feature_col[idx], color="cyan", marker="o", s=18, alpha=0.8,
                       label="Injected" if i == 0 else None)

        ax.axvline(split_1, color="gray", linestyle="--", alpha=0.6)
        ax.axvline(split_2, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"F{i}: {feature_names[i]}")
        if i == 0:
            ax.legend(loc="upper right")

    # remove empty axes
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    plt.savefig(figure_path, bbox_inches="tight", dpi=150)
    plt.close()
    
def plot_injected(test_clean, test_raw, t, f, tag, feature_names, path, lookback=72):
    """
    Plots a local comparison between the original and injected signals for a synthetic anomaly.
    """
    # Manejar si f es una sola feature o varias (multi-feature)
    features_to_plot = [f] if isinstance(f, (int, np.integer)) else f
    
    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(10, 4 * len(features_to_plot)), sharex=True)
    if len(features_to_plot) == 1: axes = [axes]
    
    start, end = max(0, t- 96), min(test_raw.shape[0], t + 96)
    time_axis = np.arange(start, end)

    for i, feat_idx in enumerate(features_to_plot):
        axes[i].plot(time_axis, test_clean[start:end, feat_idx], color='black', alpha=0.7, label='Original', linewidth=1.5)
        axes[i].plot(time_axis, test_raw[start:end, feat_idx], color='red', alpha=0.8, label='Injected', linestyle='--')
        
        # Resaltar el punto/ventana exacta del error
        feat_name = feature_names[feat_idx] if feature_names is not None else f"Feature {feat_idx}"
        axes[i].set_title(f"{tag} | {feat_name} at t={t}")
        axes[i].legend()

    plt.tight_layout()
    save_path = f"{path}/injected_{t}_{tag.replace(' ', '_')}_window{t-lookback}.png"
    plt.savefig(save_path)
    plt.close()

def save_adj_plots(A, exp_path, step=0, set="train"):
    """
    Saves a heatmap of the learned feature correlation or adjacency matrix.
    A_batch: [B, F, F]
    """
    os.makedirs(exp_path, exist_ok=True)
    
    A_plot = np.where(np.abs(A) >= 0.5, A, 0.0)
    plt.figure(figsize=(5, 4))
    plt.imshow(A_plot, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()

    plt.title(f"Correlation Batch {step} ({set.capitalize()})")

    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f"corr_B{step}_{set}.png"))
    plt.close()

def save_train_metrics_plot(metrics, figure_path):
    '''
    Plot train and validation losses per epoch and save the image
    '''
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['train_loss'], label='Train loss')
    plt.plot(metrics['validation_loss'], label='Val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

def save_component_train_metrics_plot(metrics, figure_path):
    '''
    Plot train and validation losses per epoch and save the image
    '''
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['train_loss'], label='Train loss')
    plt.plot(metrics['validation_loss'], label='Val loss')
    plt.plot(metrics['train_loss_cl'], label='Cl Train loss')
    plt.plot(metrics['validation_loss_cl'], label='Cl Val loss')
    plt.plot(metrics['train_loss_reg'], label='Reg Train loss')
    plt.plot(metrics['validation_loss_reg'], label='Reg Val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

def hist_anom(scores, threshold, figure_path, sequence_tracker=None):
    '''
    Plot a histogram of anomaly scores with optional bin coloring based on ground truth.
    '''
    scores = np.asarray(scores).flatten()

    # Histogram bin setup
    bins = np.linspace(scores.min(), scores.max(), 31)
    bin_indices = np.digitize(scores, bins) - 1
    bin_counts = np.zeros(len(bins) - 1)

    if sequence_tracker is not None:
        sequence_tracker = np.asarray(sequence_tracker)
        bin_anomalies = np.zeros(len(bins) - 1)

        # Count instances and anomalies per bin
        for idx, bin_idx in enumerate(bin_indices):
            bin_idx = int(bin_idx)
            if 0 <= bin_idx < len(bin_counts):
                bin_counts[bin_idx] += 1
                bin_anomalies[bin_idx] += sequence_tracker[idx]

        # Color blending
        bar_colors = []
        for a, c in zip(bin_anomalies, bin_counts):
            if c == 0:
                bar_colors.append((0.7, 0.7, 0.7))  # gray for empty bins
            else:
                anomaly_ratio = a / c
                normal_ratio = 1 - anomaly_ratio
                color = (
                    normal_ratio * to_rgba('blue')[0] + anomaly_ratio * to_rgba('red')[0],
                    normal_ratio * to_rgba('blue')[1] + anomaly_ratio * to_rgba('red')[1],
                    normal_ratio * to_rgba('blue')[2] + anomaly_ratio * to_rgba('red')[2],
                    1.0
                )
                bar_colors.append(color)
    else:
        # Simple gray bars if no ground truth is provided
        for idx in bin_indices:
            if 0 <= idx < len(bin_counts):
                bin_counts[idx] += 1
        bar_colors = ['gray'] * len(bin_counts)

    # Plot
    plt.figure(figsize=(10, 6))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, bin_counts, width=(bins[1] - bins[0]), color=bar_colors, align='center')
    plt.axvline(threshold, color='black', linestyle='dashed', label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Anomaly score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    
def permutation_importance(scores, model, figure_path, data_path, forecast_horizon, lookback):
    '''
    Estimates and plots feature importance using permutation-based analysis.
    '''
    baseline_score = scores 

    full_data= pd.read_csv(data_path)
    full_data = full_data.iloc[:, 1:] 
    feature_names = pd.read_csv(data_path, nrows=0).columns.tolist()
    feature_names = feature_names[1:] #datetime
    n_features = len(feature_names)
    
    #define test set data 
    lensc=len(scores)
    lent=lensc+forecast_horizon
    data_test= full_data.iloc[-lent:]
    final_size= len(data_test) - forecast_horizon - lookback + 1

    feature_importances = [] #empty list to append features importances

    for i in range(n_features):
        
        data_test_perm = data_test.copy() #define another data_test as copy of the original one
        data_test_perm.iloc[:,i] = 0  # Remove the i-th feature by setting it to zero
        perm_scores = []
        
        for idx in range(final_size):
            
            x = data_test_perm.iloc[idx:idx + lookback, :]
            y = data_test_perm.iloc[idx + lookback:idx + lookback + forecast_horizon, :]
            
            #convert x and y to tensors to use the model
            x = x.apply(pd.to_numeric) 
            x = x.fillna(0) 
            x_np = x.to_numpy().astype('float32')
            y_np = y.apply(pd.to_numeric)
            y = y.fillna(0)
            y_np = y.to_numpy().astype('float32')
            x= torch.tensor(x_np, dtype=torch.float32) 
            y= torch.tensor(y_np, dtype=torch.float32)

            #send x and y to cuda and add B dimension 
            device = torch.device("cuda")
            x, y = x.to(device), y.to(device) #as model is in gpu
            x=x.unsqueeze(0) # add B dimension at position 0 to T, F --> B, T, F; B=1
            y=y.unsqueeze(0)
            
            '''
            unsqueeze() "adds" a superficial 1 dimension to tensor (at the specified dimension: 0, 1, 2...),
                unsqueeze(0): 1, T, F; unqueeze(1): T, 1, F; unsqueeze(2) T, F, 1
             
            while squeeze removes all superficial 1 dimensions from tensor.
                squeeze: 1, T, F --> T, F
            '''
            
            y_pred = model(x, y, y, 0.0)  # Teacher forcing = 0
            loss = torch.mean((y_pred - y) ** 2).item()
            perm_scores.append(loss)

        # higher differenteces between the mean scores with and without the feature 
        # indicates higher importance of feature in model performance
        importance = np.abs(np.mean(baseline_score) - np.mean(perm_scores))
        feature_importances.append(importance)
        
    feature_importances = np.array(feature_importances)
    # Plot feature importance
    plt.figure(figsize=(10, 15))
    plt.barh(feature_names, feature_importances, color="skyblue", alpha=0.7)
    plt.xlabel("Feature Importance (Change in Anomaly Score)")
    plt.ylabel("Feature Index")
    plt.title("Permutation Feature Importance")
    plt.gca().invert_yaxis() #horizontal barplot
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

def plot_top_feature_differences(feature_differences_all, anomalies_test, figure_path, top_k=5):
    " Plots the top contributing features for each detected anomaly."
    if not isinstance(anomalies_test, torch.Tensor):
        anomalies_test = torch.tensor(anomalies_test)
    
    anomaly_indices = torch.where(anomalies_test)[0]
    num_anomalies = len(anomaly_indices)

    # Layout for subplots
    cols = 4
    rows = math.ceil(num_anomalies / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    for i, idx in enumerate(anomaly_indices):
        diffs = feature_differences_all[idx]
        top_diffs, top_idxs = torch.topk(diffs, top_k)

        ax = axs[i]
        ax.bar(range(top_k), top_diffs.numpy(), tick_label=[f"F{j.item()+1}" for j in top_idxs])
        ax.set_title(f"Anomaly {i+1} (Idx {idx.item()})")
        ax.set_ylabel("Diff")
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(len(anomaly_indices), len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()

def remove_folder(path):
    folder = Path(path)
    if folder.exists() and folder.is_dir():
        for item in folder.rglob('*'):
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        folder.rmdir()
                
def plot_test_scores(test_scores, threshold=None, figure_path=None, ek_mask=None):
    """
    Plots anomaly scores over time, highlighting detected anomalies, EK events, and the decision threshold.
    """
    test_scores = np.asarray(test_scores).ravel()
    indices = np.arange(len(test_scores))

    # default ek_mask to zeros if not given
    if ek_mask is None:
        ek_mask = np.zeros(len(test_scores), dtype=bool)
    else: 
        ek_mask = ek_mask[-len(test_scores):] # lb

    normal_indices = indices[test_scores < threshold]
    anomaly_indices = indices[test_scores >= threshold]

    plt.figure(figsize=(8,4))
    plt.scatter(normal_indices, test_scores[normal_indices], color='royalblue', label='Predicted normal', s=10)
    plt.scatter(anomaly_indices, test_scores[anomaly_indices], color='red', label='Predicted anomaly', s=15)
    
    ek_mask = (ek_mask > 0.5).astype(int) #for probabilistic
    ek_indices = indices[ek_mask.astype(bool)]
    plt.scatter(ek_indices, test_scores[ek_mask.astype(bool)], 
                facecolors='none', edgecolors='gold', s=50, label='EK=1')

    if threshold is not None:
        plt.axhline(threshold, color='orange', linestyle='--', label=f'Threshold = {threshold:.4f}')

    plt.xlabel('Index')
    plt.ylabel('Test score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def plot_attention_evolution(attention_history, figure_path):
    " Visualizes the evolution of feature attention weights throughout training."
    plt.figure(figsize=(14, 8))
    epochs = sorted(attention_history.keys())
    data = np.stack([attention_history[e] for e in epochs])  # Shape: [n_epochs, n_features
    colors = plt.cm.hsv(np.linspace(0, 1, data.shape[1]))
    
    for feat_idx in range(data.shape[1]):
        plt.plot(epochs, data[:, feat_idx],
                color=colors[feat_idx],
                alpha=0.7,
                linewidth=1)
    
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=f'Feature {i}')
        for i in range(data.shape[1])]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=2,
        fontsize='small',
        title='Features'
    )

    plt.xlabel("Epoch")
    plt.ylabel("Attention Weights")
    plt.title("Attention weights per Feature over Epochs")
    plt.tight_layout()    
    plt.savefig(figure_path)
    plt.close()

def plot_anomaly_distributions(combined, rec_diff, fc_diff, labels, figure_path, bins=20):
    "Compares the distributions of reconstruction, forecasting, and combined anomaly scores for normal and anomalous samples."
    combined=np.array(combined)
    rec_diff = np.array(rec_diff)
    fc_diff = np.array(fc_diff)
    labels   = np.array(labels)


    def split(x):
        return x[labels == 0], x[labels == 1]

    r_n, r_a = split(rec_diff)
    f_n, f_a = split(fc_diff)
    c_n, c_a = split(combined)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Reconstruction residual ---
    axs[0].hist(r_n, bins=bins, alpha=0.6, label="Real data", density=True)
    axs[0].hist(r_a, bins=bins, alpha=0.6, label="Injected Anomaly", density=True)
    axs[0].set_title("Rec score")
    axs[0].legend()

    # --- FC residual ---
    axs[1].hist(f_n, bins=bins, alpha=0.6, label="Real data", density=True)
    axs[1].hist(f_a, bins=bins, alpha=0.6, label="Injected Anomaly", density=True)
    axs[1].set_title("FC score")
    axs[1].legend()

    # --- Combined ---
    axs[2].hist(c_n, bins=bins, alpha=0.6, label="Real data", density=True)
    axs[2].hist(c_a, bins=bins, alpha=0.6, label="Injected Anomaly", density=True)
    axs[2].set_title("Combined score")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
   
def plot_instance(y_rec_pred, y_fc_pred, y_rec, y_fc, idx, ctx,feat_contrib):
    " Visualizes reconstruction and forecasting predictions for the most influential features of a selected anomaly."

    y_gt = torch.cat([y_rec, y_fc.unsqueeze(1)], dim=1)  # [B, T+1, F]

    err_rec = (y_rec_pred - y_rec).abs().mean(dim=(0, 1))  # [F]
    err_fc  = (y_fc_pred - y_fc).abs().mean(dim=0)         # [F]
    top_rec = torch.topk(err_rec, k=5).indices
    top_fc  = torch.topk(err_fc, k=5).indices

    T_rec = y_rec.shape[1]
    T_tot = T_rec + 1

    save_dir = os.path.join(ctx["experiment_path"], "high_score_preds")
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, f"pred_window_{idx}.png")

    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    
    featc = feat_contrib[idx]

    top_feat_ids = np.argsort(featc)[::-1][:7]
    top_feat_vals = featc[top_feat_ids]

    title_txt = "Top contrib features: " + ", ".join(
        [f"F{f} ({v:.3f})" for f, v in zip(top_feat_ids, top_feat_vals)]
    )

    fig.suptitle(title_txt, fontsize=16, fontweight="bold")

    x = list(range(idx, idx + T_tot))

    for i, f in enumerate(top_rec):
        ax = axes[0, i]
        gt = y_gt[0, :, f].detach().cpu().numpy()
        rec_pred = y_rec_pred[0, :, f].detach().cpu().numpy()
        ax.plot(x, gt, color="black", label="Real signal")
        ax.plot(x[:T_rec],rec_pred,color="orangered",label="Pred REC")
        ax.scatter([x[T_rec]],[y_fc_pred[0, f].detach().cpu().numpy()],color="darkcyan",label="Pred FC")
        ax.set_title(f"F{f.item()} - REC error")
        ax.legend()

    for i, f in enumerate(top_fc):
        ax = axes[1, i]
        gt = y_gt[0, :, f].detach().cpu().numpy()
        rec_pred = y_rec_pred[0, :, f].detach().cpu().numpy()
        ax.plot(x, gt, color="black", label="Real signal")
        ax.plot( x[:T_rec], rec_pred, color="orangered", label="Pred REC")
        ax.scatter([x[T_rec]], [y_fc_pred[0, f].detach().cpu().numpy()], color="darkcyan", label="Pred FC" )
        ax.set_title(f"F{f.item()} - FC error")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(base_path)
    plt.close()

def plot_test_timeline(x, y_rec, y_fc, anomaly_index, save_path):
    "Reconstructs the temporal evolution of predictions and forecasts across the test set, highlighting anomalous regions."
    if isinstance(x, list):
        x = torch.cat(x, dim=0)
    if isinstance(y_rec, list):
        y_rec = torch.cat(y_rec, dim=0)
    if isinstance(y_fc, list):
        y_fc = torch.cat(y_fc, dim=0)

    x = x.cpu().numpy()
    y_rec = y_rec.cpu().numpy()
    y_fc = y_fc.cpu().numpy()

    N, T, F = y_rec.shape

    # ---------------------------------------------------
    # TAKE ONLY LAST TIME STEPS (TAIL REPRESENTATIONS)
    # ---------------------------------------------------
    x_last = x[:, -1, :]        # (N, F)
    rec_last = y_rec[:, -1, :]  # (N, F)

    # rec2 = second-to-last timestep inside each window
    rec2 = y_rec[:, -2, :] if T >= 2 else np.full_like(rec_last, np.nan)

    # rec3 = third-to-last timestep
    rec3 = y_rec[:, -3, :] if T >= 3 else np.full_like(rec_last, np.nan)

    # forecast already per window
    fc_last = y_fc[:, -1, :] if y_fc.ndim == 3 else y_fc  # (N, F)
    
    
    anomaly_index = np.asarray(anomaly_index).astype(int)

    # Find contiguous anomaly regions
    anomaly_segments = []
    in_segment = False

    for i, val in enumerate(anomaly_index):
        if val == 1 and not in_segment:
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            anomaly_segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        anomaly_segments.append((start, len(anomaly_index) - 1))
        

    # ---------------------------------------------------
    # CONCAT ACROSS N → TIME AXIS
    # ---------------------------------------------------
    t = np.arange(N)

    ncols = 4
    nrows = math.ceil(F / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 3 * nrows)
    )
    axes = axes.flatten()

    for f in range(F):
        
        ax = axes[f]

        ax.plot(t, x_last[:, f],
                color="black", linewidth=1.5, label="x (last)")

        ax.plot(t, rec_last[:, f],
                color="green", linewidth=1.0, alpha=0.5, label="rec1 (last)" )

        ax.plot(t, rec2[:, f],
                color="orange", linewidth=1.0, alpha=0.5, label="rec2 (t-1)")

        ax.plot(t, rec3[:, f],
                color="red", linewidth=1.0, alpha=0.5, label="rec3 (t-2)")

        ax.plot(t+1, fc_last[:, f],
                color="blue", linewidth=1.0, alpha=0.5, label="forecast")
        
        for start, end in anomaly_segments:
            ax.axvspan(start - 0.5,end + 0.5,color="yellow",alpha=0.4, zorder=0 )

        ax.set_title(f"Feature {f}")

    for j in range(F, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_error_diagnostics(rec_err_raw_all, fc_err_raw_all,rec_norm_all, fc_norm_all,feature_names=None, anomaly_ranges=None,
                                    rho=0.05, ncols=3,figsize_per_subplot=(4.5, 3.2),save_path=None):

    "Visualizes raw and normalized reconstruction and forecasting errors for every feature."
    rec_err_cat  = torch.cat(rec_err_raw_all, dim=0).numpy()[:, -1, :]  # [N,F] last-step raw
    fc_err_cat   = torch.cat(fc_err_raw_all, dim=0).numpy()             # [N,F]
    rec_norm_cat = torch.cat(rec_norm_all, dim=0).numpy()[:, -1, :]     # [N,F] last-step normalized
    fc_norm_cat  = torch.cat(fc_norm_all, dim=0).numpy()                # [N,F]

    N, F = rec_err_cat.shape
    if feature_names is None:
        feature_names = [f"F{f}" for f in range(F)]

    ncols = min(ncols, F)
    nrows = int(np.ceil(F / ncols))
    x = np.arange(N)

    fig, axes = plt.subplots(nrows * 2, ncols,
                              figsize=(figsize_per_subplot[0] * ncols,
                                       figsize_per_subplot[1] * nrows),
                              squeeze=False)

    norm_line = 1 - rho  # e.g. 0.95: the "edge of normal" in normalized space

    for f in range(F):
        row_block = (f // ncols) * 2
        col = f % ncols

        # --- top: raw error ---
        ax_raw = axes[row_block][col]
        ax_raw.plot(x, rec_err_cat[:, f], lw=0.6, color="tab:blue", label="rec (raw)")
        ax_raw.plot(x, fc_err_cat[:, f], lw=0.6, color="tab:orange", alpha=0.7, label="fc (raw)")
        raw_ref = np.percentile(rec_err_cat[:, f], 100 * norm_line)
        ax_raw.axhline(raw_ref, color="grey", lw=0.8, ls="--")

        # --- bottom: normalized error ---
        ax_norm = axes[row_block + 1][col]
        ax_norm.plot(x, rec_norm_cat[:, f], lw=0.6, color="tab:blue", label="rec (norm)")
        ax_norm.plot(x, fc_norm_cat[:, f], lw=0.6, color="tab:orange", alpha=0.7, label="fc (norm)")
        ax_norm.axhline(norm_line, color="grey", lw=0.8, ls="--")

        if anomaly_ranges is not None:
            for (s, e) in anomaly_ranges:
                ax_raw.axvspan(s, e, color="red", alpha=0.15, lw=0)
                ax_norm.axvspan(s, e, color="red", alpha=0.15, lw=0)

        ax_raw.set_title(f"{feature_names[f]} (raw)", fontsize=8)
        ax_norm.set_title(f"{feature_names[f]} (normalized)", fontsize=7, color="dimgray")
        ax_raw.tick_params(labelsize=6)
        ax_norm.tick_params(labelsize=6)

        if f == 0:
            ax_raw.legend(fontsize=6, loc="upper right")
            ax_norm.legend(fontsize=6, loc="upper right")

    # blank out unused axes
    total_rows = nrows * 2
    for f in range(F, nrows * ncols):
        row_block = (f // ncols) * 2
        col = f % ncols
        axes[row_block][col].axis("off")
        axes[row_block + 1][col].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig