#!/bin/bash

DIR="$(dirname "$0")"
model_path="/root/research/projects/ByProt/run/logs/fixedbb/cath_4.2/lm_design_esm2_650m"
# model_path="/root/research/projects/ByProt/run/logs/fixedbb/cath_4.3/lm_design_esm2_650m_gvptrans"
# model_path="/root/research/projects/ByProt_public/logs/fixedbb_multichain/lm_design_esm2_650m"

temperature=0.1
pdb_dir="/root/research/projects/ByProt/data/pdb_samples"
out_dir="$pdb_dir/lm_design_fasta"

python $DIR/design_pdb.py \
    --experiment_path $model_path --ckpt "best.ckpt" \
    --pdb_dir $pdb_dir --out_dir $out_dir \
    --seed 42 \
    --num_seqs 1 \
    --temperature $temperature \
    --max_iter 5
