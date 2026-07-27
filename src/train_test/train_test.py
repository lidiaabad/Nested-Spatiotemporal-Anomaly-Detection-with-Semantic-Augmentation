import os, time, torch, sys, json
import numpy as np
import torch.nn as nn
import scipy.stats as stats
import torch.nn.functional as F
from config.arguments import parse_arguments
from utils import helpers_setup as hs
from utils import helpers_plots as hp
from utils import helpers_output as ho
from models import archs as models
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader
from data.pilots_preprocessing import TSDataset
from copy import deepcopy
from collections import defaultdict

# HELPERS: LOSS, MODEL INITIALIZATION AND INFORMATION PRINTING FOR DEBUG
class TrimmedMaskedMAELoss(nn.Module):
    "Loss function: TRimmed MAE for low contamination"
    def __init__(self, trim_frac=0.05):
        super().__init__()
        self.trim_frac = trim_frac

    def forward(self, y_pred, y_true, mask=None):
        loss = torch.abs(y_pred - y_true)   # [B, T, F] or [B, F]

        if mask is not None:
            loss = loss * mask

        if loss.dim() == 3:
            B, T, F = loss.shape
            # trim per-feature over the B*T dimension
            loss_flat = loss.permute(2, 0, 1).reshape(F, -1)   # [F, B*T]
            if mask is not None:
                mask_flat = mask.permute(2, 0, 1).reshape(F, -1)
            k = max(1, int((1 - self.trim_frac) * loss_flat.shape[1]))
            # keep lowest k errors per feature
            kept, _ = torch.topk(-loss_flat, k, dim=1)         # [F, k]
            if mask is not None:
                # weight by alive ratio per feature
                alive_ratio = mask_flat.float().mean(dim=1)     # [F]
                per_feat = (-kept).mean(dim=1)                  # [F]
                return (per_feat * alive_ratio).sum() / (alive_ratio.sum() + 1e-8)
            return (-kept).mean()

        elif loss.dim() == 2:
            B, F = loss.shape
            loss_t = loss.T                                      # [F, B]
            k = max(1, int((1 - self.trim_frac) * B))
            kept, _ = torch.topk(-loss_t, k, dim=1)
            if mask is not None:
                alive_ratio = mask.float().mean(dim=0)          # [F]
                per_feat = (-kept).mean(dim=1)
                return (per_feat * alive_ratio).sum() / (alive_ratio.sum() + 1e-8)
            return (-kept).mean()

        raise ValueError(f"Unexpected shape: {loss.shape}")
    
def build_optimizer(model, args):
    if args.optimizer_pt.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay_pt)
    if args.optimizer_pt.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay_pt)
    return torch.optim.SGD(model.parameters(),lr=args.lr_pt,momentum=args.momentum_pt,weight_decay=args.weight_decay_pt)

def initialize_weights(model, args):
    init_type = args.weight_init_pt.lower() if args.weight_init_pt else "xavier"
    #print(f"Initializing weights according to {init_type}")
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            # Iteramos por todos los parámetros por nombre (maneja num_layers > 1)
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # Inicializamos bias en 0
                    param.data.fill_(0)
                    # TRUCO CRÍTICO: Forget gate bias = 1
                    # En PyTorch, los bias de LSTM están concatenados (ingate, forget, cell, outgate)
                    # El forget gate es el segundo cuarto del vector de bias.
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
                    
        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)  # pequeño positivo

def build_model(args, ctx):
    "Build model getting data dim"
    for x, _, _, _, _  in ctx["train_loader"]:
        data_dim = x.shape[-1]
        input_shape=x.shape
        break
    model = models.SerializableModule().create(args.arc_pt,data_dim=data_dim,hidden_dim=args.hidden_units_pt,
        dropout=args.dropout_pt, activation=args.activation_pt.lower(),norm=args.norm_pt).to(ctx["device"])

    initialize_weights(model, args)

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

    window = torch.hann_window(args.fc, periodic=False).to(ctx["device"]).view(1, args.fc, 1)

    return model, optimizer, scheduler, window

def diagnose_dead_layers(model):
    ""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            stats[name] = param.grad.abs().mean().item()
    return stats

def pretty_print_mean_stats(mean_stats, threshold=1e-4):
    "Info every x epochs"
    dead = []
    alive = []

    for name, g in mean_stats.items():
        if g < threshold:
            dead.append((name, g))
        else:
            alive.append((name, g))

    report = []
    report.append("\n" + "="*50)
    report.append(f"DEAD layers (grad < {threshold}): {len(dead)}/{len(dead)+len(alive)}")
    for name, g in dead:
        report.append(f"  ✗ {name:<40} {g:.2e}")
    report.append("ALIVE layers:")
    for name, g in alive:
        report.append(f"  ✓ {name:<40} {g:.2e}")
    report.append("="*50 + "\n")

    return "\n".join(report)


# TRAIN AND VAL MODEL / PRETRAIN MODEL + TEST
def train_and_validate(model, optimizer, scheduler, window, args, ctx):
    
    train_loader = ctx["train_loader"]
    val_loader = ctx["val_loader"]
    device = ctx["device"]
    model_path = ctx["model_path"]
    feature_names = ctx.get("feature_names", None)
    beta=args.beta

    train_losses, validation_losses, epoch_times = [], [], []
    best_val_loss = np.inf
    epochs_without_improvement = 0

    criterion = TrimmedMaskedMAELoss(trim_frac=0.05)
    
    print("Starting training and validation...")
    block_attn, block_tcn=False, False
    if "noattn" in ctx["experiment_path"]: 
        block_attn=True
        print("\t with blocked feature path!!!!")
    elif "notcn" in ctx["experiment_path"]: 
        block_tcn=True
        print("\t with blocked temporal path!!!!")
    else: 
        print("\t with both paths")
    
    if "norevin" in ctx["experiment_path"]: 
        revin=False
        print("\t deactivated revin")
    else: 
        revin=True
        print("\t activated revin")
        
  
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss_epoch = 0.0
        val_loss_epoch = 0.0
        
        grad_stats_epoch = []

        # ==================== TRAIN ====================
        model.train()

        for batch_idx, (x, y_rec, y_fc, alive, label) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(device)
            y_rec = y_rec.to(device)
            y_fc = y_fc.to(device)
            alive = alive.to(device)
            alive_rec=alive[:, :-1, :]
            alive_fc=alive[:, -1,: ]
            label = label.to(device)
            
            y_rec_pred, y_fc_pred= model(x, block_attn, block_tcn, revin) 
            
            loss_rec = criterion(y_rec_pred, y_rec, alive_rec) 
            loss_fc = criterion(y_fc_pred, y_fc, alive_fc)
            loss = beta* loss_rec + (1-beta)*loss_fc 
            loss.backward()
            
            if epoch % args.epochsinfo == 0 or (epoch < 50 and epoch % 5 == 0):
                stats = diagnose_dead_layers(model)
                grad_stats_epoch.append(stats)
            optimizer.step()
            train_loss_epoch += loss.item()
            
        # ==================== VALIDATION ====================
        model.eval()
        with torch.no_grad():
            for batch_idx_val, (x, y_rec, y_fc, alive, label) in enumerate(val_loader):

                x = x.to(device)
                y_rec = y_rec.to(device)
                y_fc = y_fc.to(device)
                alive = alive.to(device)
                alive_rec=alive[:, :-1, :]
                alive_fc=alive[:, -1,: ]
                label = label.to(device)

                y_rec_pred, y_fc_pred= model(x, block_attn, block_tcn, revin)
            
                val_loss_rec = criterion(y_rec_pred, y_rec, alive_rec) 
                val_loss_fc  = criterion(y_fc_pred, y_fc, alive_fc)
                val_loss = beta*val_loss_rec + (1-beta)*val_loss_fc
                
                if epochs_without_improvement == (args.patience - 1) and batch_idx_val == 0: 
                    print(f"FINAL Losses: rec - {val_loss_rec.mean():.4f}; fc - {val_loss_fc.mean(): .4f}; all - {val_loss.mean():.4f}")

                val_loss_epoch += val_loss.item()

        # ==================== BOOKKEEPING ====================
        train_loss_epoch /= len(train_loader)
        val_loss_epoch /= len(val_loader)
        train_losses.append(train_loss_epoch)
        validation_losses.append(val_loss_epoch)
        epoch_times.append(time.time() - start_time)

        scheduler.step(val_loss_epoch)

        # ---- EARLY STOPPING ----
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_without_improvement = 0
            model.save_entire_model(model_path)
            model.save(model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement % 5 == 0:
                print(f"{epochs_without_improvement} epochs without improvement")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # ==================== DEBUG PRINT ====================
        if epoch % args.epochsinfo == 0: 
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{args.epochs} | "
                  f"Time {time.time()-start_time:.2f}s | "
                  f"Train {train_loss_epoch:.4f} | Val {val_loss_epoch:.4f} | LR {lr:.2e}")
            
            mean_stats = {}
            for stats in grad_stats_epoch:
                for k, v in stats.items():
                    mean_stats.setdefault(k, []).append(v)

            mean_stats = {k: np.mean(vs) for k, vs in mean_stats.items()}
            print(pretty_print_mean_stats(mean_stats))

    return { "train_loss": train_losses, "validation_loss": validation_losses, "epoch_times": epoch_times}

def load_pretrained_model(model, model_path, device):
    pt_path = f"{model_path}.pt"
    model.load_state_dict(torch.load(pt_path, weights_only=True, map_location=device), strict=False)
    return model

def test_model(model, args, ctx, loader):
    
    beta=args.beta
    device = ctx["device"]
    
    rec_mean_list, fc_mean_list, all_scores=[], [], []
    rec_trajs, fc_preds, x_hist = [], [], []

    criterion=nn.L1Loss(reduction="none") 
    
    print("Starting testing...")
    
    block_attn = "noattn" in ctx["experiment_path"]
    block_tcn  = "notcn"  in ctx["experiment_path"]
    revin      = "norevin" not in ctx["experiment_path"]
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y_rec, y_fc, alive, label) in enumerate(loader):
            
            x = x.to(device)
            y_rec = y_rec.to(device)
            y_fc = y_fc.to(device)
            alive = alive.to(device)
            alive=alive[:, :-1, :]
            
            y_rec_pred, y_fc_pred = model(x, block_attn, block_tcn,revin=revin)
                
            loss_rec = criterion(y_rec_pred, y_rec)
            loss_fc = criterion(y_fc_pred, y_fc)
            loss_fc = loss_fc.unsqueeze(1).expand_as(loss_rec)         # [B, T, F]

            loss = beta*loss_rec + (1-beta)* loss_fc 
            
            inst_score = loss.mean(dim=(1, 2)) if loss.dim() == 3 else loss.mean(dim=1)
            all_scores.append(inst_score.cpu())

            rec_trajs.append(y_rec_pred.cpu())
            fc_preds.append(y_fc_pred.cpu())
            x_hist.append(y_rec.cpu())

            rec_mean_list.append(loss_rec.mean().item())
            fc_mean_list.append(loss_fc.mean().item())

    scores = torch.cat(all_scores).numpy()
    rec_mean = rec_mean_list
    fc_mean  = fc_mean_list
            
    hp.plot_test_timeline(x_hist, rec_trajs,fc_preds, ctx["injected_anomaly_index"], save_path=os.path.join(ctx["experiment_path"], "timeline_test.png"))
    return {
        "scores":    scores,
        "rec_mean":  rec_mean,
        "fc_mean":   fc_mean,
    }


# OUTPUTS: METRICS AND SCORES
def compute_metrics(scores, labels, K_ratio=0.1, len_inj_anom=None):
    labels = np.asarray(labels).astype(int)
    print("Using len for PA: ", len_inj_anom)

    if labels.sum() == 0:
        return {"AP": np.nan, "P@K": np.nan, "R@K": np.nan}

    ap = average_precision_score(labels, scores)
    K = max(1, int(K_ratio * len(scores)))
    topk = np.argsort(-scores)[:K]
    p_at_k = labels[topk].sum() / K
    r_at_k = labels[topk].sum() / labels.sum()

    # Point-adjusted Recall
    if len_inj_anom is not None and len_inj_anom > 1:

        labels_pa = np.zeros_like(labels)
        starts = np.where((labels == 1) & (np.concatenate([[0], labels[:-1]]) == 0))[0]
        ends = np.where((labels == 1) & (np.concatenate([labels[1:], [0]]) == 0))[0]

        topk_mask = np.zeros_like(labels)
        topk_mask[topk] = 1

        for s, e in zip(starts, ends):
            if topk_mask[s:e+1].any():
                labels_pa[s:e+1] = 1
            else:
                labels_pa[s:e+1] = 0  
        
        den = labels_pa.sum()
        if den > 0:
            r_at_k_pa = labels_pa[topk].sum() / den
        else:
            print("No detected in topk")
            r_at_k_pa = 0.0

    else:
        r_at_k_pa = r_at_k
        
    print(f"AP {ap:.3f} | P@K {p_at_k:.3f} | R@K {r_at_k:.3f} |  R@K_PA {r_at_k_pa:.3f} | K used {K}")
    
    return {"AP": ap,"P@K": p_at_k,"R@K": r_at_k,"R@K_PA": r_at_k_pa,}

def build_scores(train_scores, val_scores, test_scores, lb=36, split_order=None):
    if split_order is None:
        split_order = ["train", "val", "test"]
    pad=lb-1
    scores_dict = {"train": train_scores, "val": val_scores, "test": test_scores}

    concatenated_scores= []
    for split in split_order:
        scores = scores_dict[split]
        concatenated_scores.append(np.concatenate([np.zeros(pad), scores]))
        
    return np.concatenate(concatenated_scores)


def run_pipeline(ctx, ek_rules, dataset):
    
    args = parse_arguments()
    model, optimizer, scheduler, window = build_model(args, ctx)
    reporter = ctx["reporter"]
    experiment_path = ctx["experiment_path"]

    # -----------------1a: TRAIN + VALIDATE MODEL-------------------------
    # -----------------1b: LOAD PRETRAINED MODEL-------------------------
    train_metrics = None
    if not args.eval_only_pt:
        train_metrics = train_and_validate(model=model,optimizer=optimizer,scheduler=scheduler,window=window,args=args,ctx=ctx)
        reporter.report("train_metrics", train_metrics)
        hp.save_train_metrics_plot(train_metrics, os.path.join(experiment_path, "train_metrics.png"),)    
    else:
        model = load_pretrained_model(model=model, model_path=ctx["loaded_path"],device=ctx["device"])
    
    # ----------- 2: COMPUTE TRAIN AND VAL SCORES AND THRESHOLD ----------------
    scale=dataset.feat_scale
    scores_train, scores_val = ho.get_all_scores(model, ctx, ctx["train_loader"], ctx["val_loader"], beta=args.beta, device=ctx["device"], return_test=False, scale=scale)
    scores_train_val = np.concatenate([scores_train, scores_val])
    scores_train_val = scores_train_val[np.isfinite(scores_train_val)]
    threshold = ho.threshold_computation(scores_train_val, threshold_method=args.threshold_method)
    print("FINAL THRESHOLD:", threshold)
    # for sigmoid 
    tail = scores_train_val[scores_train_val>threshold]
    tau = tail.std()
    
    # --------------- 3: EVALUATION ------------------
    test_loader=ctx["test_loader"]
    injected_labels=ctx["injected_anomaly_index"]
    feature_names=ctx["feature_names"] 
    test_output=test_model(model, args, ctx, test_loader)
    scores_test, feats_test, rec_test, fc_test = ho.get_all_scores(model, ctx, train_loader=ctx["train_loader"], val_loader=ctx["val_loader"], test_loader=test_loader, beta=args.beta, device=ctx["device"], return_test=True, scale=scale)
    
    len_inj_anom = {
        "point": 1,
        "contextual": 3,
    }.get(args.anom_type, 12) # 12 for collective
    results = compute_metrics(scores_test, injected_labels, K_ratio=0.1, len_inj_anom=len_inj_anom)
    reporter.report("injected_metrics", {"AP":   results["AP"],"P@K":  results["P@K"],"R@K": results["R@K"]})

    # --------------- 4: OUTPUT AND PLOTS ------------------
    feature_names = ctx.get("feature_names", None)
    resp = ho.get_info_anomalies(feats_test, scores_test, threshold, feature_names, tau) 
    reporter.report("test_anomalies", resp)

    hp.hist_anom(scores_train_val, threshold, os.path.join(ctx["experiment_path"], "anom_hist_trainval.png"))
    hp.hist_anom(scores_test, threshold, os.path.join(ctx["experiment_path"], "anom_hist_test.png"))
    hp.plot_anomaly_distributions(scores_test, rec_test, fc_test, injected_labels, os.path.join(ctx["experiment_path"], "scores_dist_inj_anoms.png"))
    
    args.split_order = [s.strip() for s in args.split_order.split(',')]
    scores = build_scores(scores_train, scores_val, scores_test, lb=args.lb, split_order=args.split_order)
    hp.plot_test_scores(scores, threshold, os.path.join(ctx["experiment_path"], "all_scores.png"), ek_rules)
    hp.plot_test_scores(scores_test, threshold, os.path.join(ctx["experiment_path"], "test_scores.png"), ek_rules)
    
    reporter.save()
    return results
