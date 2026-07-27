import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='ARGUS arguments')

    # --- Core experiment arguments ---
    parser.add_argument('--mode', type=str, help='Mode: unsupervised, supervised-real or supervised-synthetic', required=True)
    parser.add_argument('--mode_pt', type=str, help='Mode: reconstruction, forecast, or both ', required=True)
    parser.add_argument('--data_path', type=str, help='path to the folder containing the ARGUS data', required=True)
    parser.add_argument('--dataset', default="ARGUS", type=str, help="Dataset to be used: SMD, SMAP, MSL")
    parser.add_argument('--eval_only_pt', action="store_true", default=False, help='Argument to indicate that a pretrained model just need to be tested')
    parser.add_argument('--loaded_path', type=str, help='path to the pretrained model')
    
    parser.add_argument('--model_id', default='', help='Identifier of model parameter changes')
    parser.add_argument('--checkpoint', default='models/checkpoints', metavar='CHECKPOINT', help='checkpoints directory')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false', help='do not use cuda')

    parser.add_argument('--ek', action='store_true', help="Usage of EK rules: True/False")
    parser.add_argument('--targets', default='anomalies', help='downstream target: anomalies, dangers, both')

    # Generic data arguments
    parser.add_argument('--lb', type=int, default=30, help='time window for time series processing')
    parser.add_argument('--fc', type=int, default=12, help='time window for time series forecasting')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--split_order', type=str,default='train,val,test',help='Comma-separated order of splits for preprocessing and pretext scores (e.g., "test,val,train")')
    parser.add_argument('--measure_frequency', type=int, default=5, help="Frequency in minutes for measurements")

    # --- EVT / thresholding ---
    parser.add_argument('--threshold_method', default="POT", help='threshold selection method: POT or optimized_POT')
    parser.add_argument('--optimization_score', type=str, default="f1", help='Score function to optimize in p,q search')
    parser.add_argument('--percentile', type=int, default=90, help='threshold percentile [0-100] hyperparameter in EVT')
    parser.add_argument('--quantile', type=float, default=0.95, help='threshold quantile [0-1] hyperparameter in EVT')

    # --- Augmentation / Anomaly ---
    parser.add_argument('--num_augs', type=int, default=0, help='Number of augmentations')
    parser.add_argument('--augment', action='store_true', help='Augment train and val sets: True/False')
    parser.add_argument('--augment_prob', type=float, default=0.5, help='Augment probability')
    parser.add_argument('--anom_type', type=str, default="all", help="Anomaly type to be injected: {point, contextual, collective, all}")
    parser.add_argument('--anom_sev', type=float, default=0.01, help='Anomaly severity to be injected: [0,1]')
    parser.add_argument('--anom_freq', type=float, default=0.01, help='Anomaly fraction to be injected')
    
    # ---- Training ----
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--epochsinfo', type=int, default=10, metavar='N', help='number of epochs to give info')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs of no loss improvement before stop training')
    parser.add_argument('--beta', type=float, default=0.5, help='Hybrid loss parameter: beta*rec + (1-beta)*fc')

    # --- Automatically add pretext (_pt) and downstream (_ds) versions ---
        # --- Pretext / Downstream duplication list ---
    model_args_with_defaults = {
        'arc': ('LSTMAE', str),
        'num_layers': (3, int),
        'hidden_units': (48, int),
        'num_heads': (4, int),
        'dropout': (0.2, float),
        'activation': ('ReLU', str),
        'norm': (None, str),
        'weight_init': (None, str),
        'teacher_forcing_probability': (0.0, float),
        'lr': (0.001, float),
        'optimizer': ('adam', str),
        'weight_decay': (0.0001, float),
        'delta': (1.0, float),
        'robust_lambda': (0.00065, float),
        'maxr': (1.0, float),
        'bayesian': (False, bool)  # <- proper boolean
    }

    for arg_name, (default_val, arg_type) in model_args_with_defaults.items():
        
        if arg_type is bool:
            parser.add_argument(f'--{arg_name}_pt',default=default_val,help=f'{arg_name} for pretext network')
            parser.add_argument(f'--{arg_name}_ds',default=default_val,help=f'{arg_name} for downstream network')
        else:
            # int / float / str
            parser.add_argument(f'--{arg_name}_pt', type=arg_type, default=default_val, help=f'{arg_name} for pretext network')
            parser.add_argument(f'--{arg_name}_ds', type=arg_type, default=default_val, help=f'{arg_name} for downstream network')
        
    return parser.parse_args()