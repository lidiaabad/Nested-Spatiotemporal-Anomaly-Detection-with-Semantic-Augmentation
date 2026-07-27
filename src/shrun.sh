#!/bin/bash

####################################
# SLURM directives (ignored locally)
####################################
#SBATCH --job-name=train_test_gpu
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

####################################
# Environment setup
####################################
# GPU selection (local or SLURM)
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2  # hides info messages
export TF_ENABLE_ONEDNN_OPTS=0 # disables oneDNN optimizations

# Load modules only if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "Running on server SLURM (job $SLURM_JOB_ID)"
    module load rama0.3
    module load Python/3.11.3
else
    echo "Running on work station"
fi

#base configuration
seed=42
epochs=500
epochsinfo=50
batch_size=256
patience=20
lr_pt=0.001
weight_decay=0.0001
lb=72 # 1.5 day
fc=0 #10
threshold_method="median"
optimization_score="silhouette" #recall so maximizes TP and minimizes FN (allows for FP)
num_layers=1
hidden_units=32
activation=prelu
norm=layer
dropout=0.3
weight_init="kaiming"
val_ratio=0.10
test_ratio=0.10
split_order="train,val,test"
mode="unsupervised"
anom_freq=0.05
anom_sev=0.5
anom_type="contextual"
mode_pt="both"
arch_pretext="NestedAD"

#combinations to be run in series
combinations=(
    "final_june_point05 schenkenberg data/Schenkenberg/ final_june 0.5 point 0.5"
    "final_june_point1 schenkenberg data/Schenkenberg/ final_june 0.5 point 1.0"
    "final_june_point2 schenkenberg data/Schenkenberg/ final_june 0.5 point 2.0"
    "final_june_collshift05 schenkenberg data/Schenkenberg/ final_june 0.5 collective_shift 0.5"
    "final_june_collshift1 schenkenberg data/Schenkenberg/ final_june 0.5 collective_shift 1.0"
    "final_june_collshift2 schenkenberg data/Schenkenberg/ final_june 0.5 collective_shift 2.0"
    "final_june_collvar05 schenkenberg data/Schenkenberg/ final_june 0.5 collective_variance 0.5"
    "final_june_collvar1 schenkenberg data/Schenkenberg/ final_june 0.5 collective_variance 1.0"
    "final_june_collvar2 schenkenberg data/Schenkenberg/ final_june 0.5 collective_variance 2.0"
    "final_june_collfr schenkenberg data/Schenkenberg/ final_june 0.5 collective_frozen 1.0"
    "final_june_cont schenkenberg data/Schenkenberg/ final_june 0.5 contextual 1.0"
)

for combo in "${combinations[@]}"
do
    read -r model_id database data_path loaded_path beta anom_type anom_sev <<< "$combo"
    echo "************ ${model_id^^} - ${database^^} **************"

   python3 -u src/run.py \
    --mode $mode \
    --mode_pt $mode_pt \
    --dataset $database \
    --data_path $data_path \
    --loaded_path $loaded_path \
    --model_id $model_id \
    --eval_only_pt \
    --seed $seed \
    --epochs $epochs \
    --epochsinfo $epochsinfo \
    --patience $patience \
    --batch_size $batch_size \
    --lb $lb \
    --fc $fc \
    --val_ratio $val_ratio \
    --test_ratio $test_ratio \
    --split_order $split_order \
    --beta $beta \
    --threshold_method $threshold_method \
    --optimization_score $optimization_score \
    --anom_type $anom_type \
    --anom_sev $anom_sev \
    --anom_freq $anom_freq \
    --ek \
    --arc_pt $arch_pretext \
    --num_layers_pt $num_layers \
    --hidden_units_pt $hidden_units \
    --dropout_pt $dropout \
    --activation_pt $activation \
    --norm_pt $norm \
    --weight_init_pt $weight_init \
    --lr_pt $lr_pt \
    --weight_decay_pt $weight_decay
done
