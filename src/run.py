import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent #project root 
sys.path.insert(0, str(PROJECT_ROOT))
from config.arguments import parse_arguments
from utils import helpers_setup as hs
from utils import helpers_output as ho
from train_test import train_test
import data.public_preprocessing as pub_ddbb
from data import dataloaders
from utils import helpers_plots as hp
import copy, os
import warnings
import numpy as np

if __name__ == "__main__":
    # --- Ignore all warnings ---
    warnings.filterwarnings("ignore")

    args = parse_arguments()
    print(args.bayesian_pt)

    device = hs.setup_device(args)
    
    load_entity_fn = lambda _: dataloaders.load_pilot_entity(args)

    # Evaluation only: repeat 10 times with different seeds
    if args.eval_only_pt:
        base_seed = args.seed
        all_runs = []
        for run in range(10):
            current_seed = base_seed + run
            args.seed = current_seed
            hs.set_seed(current_seed)
            print("\n" + "=" * 80)
            print(f"RUN {run + 1}/10   (seed={current_seed})")
            print("=" * 80)

            # Load dataset (new anomaly injection)
            entity_ctx = load_entity_fn(entity)
            base_dataset = entity_ctx["dataset"]
            ctx = hs.setup_experiment(args, entity_ctx, device)

            #plot all features 
            global_fig_path = os.path.join(ctx["experiment_path"],f"all_feat_{args.dataset}.png", )
            hp.feature_plot( base_dataset.raw_df_with_alive,base_dataset.feature_names,global_fig_path,)

            print("****************IMP USING PROB RULES***************")

            ek_rules = base_dataset.ek_rules[0].max(axis=1)
            loaders, test_ek, feature_names, inj_anom_index, dataset = (dataloaders.build_loaders(base_dataset,args, positive=ek_rules,))
            ctx["train_loader"] = loaders["train"]
            ctx["val_loader"] = loaders["val"]
            ctx["test_loader"] = loaders["test"]
            ctx["feature_names"] = feature_names
            ctx["injected_anomaly_index"] = inj_anom_index
            ctx["model_path"] = os.path.join( ctx["experiment_path"], args.model_id )
            ctx["loaded_path"] = hs.build_loaded_path(args)


            # Evaluate pretrained checkpoint
            results = train_test.run_pipeline( ctx,ek_rules,dataset, )
            all_runs.append(results)
            
        # Summary over the 10 runs
        print("\n" + "=" * 80)
        if entity is None:
            print("SUMMARY OVER 10 RUNS")
        else:
            print(f"SUMMARY OVER 10 RUNS ({entity})")
        print("=" * 80)
        metrics = all_runs[0].keys()
        summary = {}
        for metric in metrics:

            vals = np.asarray([r[metric] for r in all_runs],dtype=float,)
            summary[f"{metric}_mean"] = vals.mean()
            summary[f"{metric}_std"] = vals.std(ddof=1)
            print( f"{metric:20s}: " f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" )

    # Train and val (single run)
    else:

        entity_ctx = load_entity_fn(entity)
        base_dataset = entity_ctx["dataset"]
        ctx = hs.setup_experiment(args, entity_ctx, device)
        
        global_fig_path = os.path.join(ctx["experiment_path"], f"all_feat_{args.dataset}.png", )
        hp.feature_plot(base_dataset.raw_df_with_alive,base_dataset.feature_names,global_fig_path)
        
        print("****************IMP USING PROB RULES***************")

        ek_rules = base_dataset.ek_rules[0].max(axis=1)
        loaders, test_ek, feature_names, inj_anom_index, dataset = (dataloaders.build_loaders( base_dataset,args, positive=ek_rules,))
        ctx["train_loader"] = loaders["train"]
        ctx["val_loader"] = loaders["val"]
        ctx["test_loader"] = loaders["test"]
        ctx["feature_names"] = feature_names
        ctx["injected_anomaly_index"] = inj_anom_index
        ctx["model_path"] = os.path.join(ctx["experiment_path"], args.model_id, )
        ctx["loaded_path"] = hs.build_loaded_path(args)

        train_test.run_pipeline( ctx, ek_rules, dataset,)
