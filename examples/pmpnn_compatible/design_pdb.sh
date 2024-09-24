#!/bin/bash

function rdebug() {
    MASTER_ADDR="$(ss | grep 2222 | head -n 1 | tr -s " " | cut -d" " -f 6 |  sed -e "s/\[\|\]:[0-9]\+$//g")"
    # pgrep ssh$ | xargs kill -9 
    # echo $MASTER_ADDR
    ssh -R 5678:localhost:5678 -N -f root@${MASTER_ADDR} -p 9000 > /dev/null 2>&1

    python3 -m debugpy --listen localhost:5678 "$@"
}


DIR="$(dirname "$0")"
cd $DIR
# model_path="/root/research/projects/ByProt/run/logs/fixedbb/cath_4.2/lm_design_esm2_650m"
# model_path="/root/research/projects/ByProt/run/logs/fixedbb/cath_4.3/lm_design_esm2_650m_gvptrans"
model_path="/root/research/projects/others/ByProt_public/logs/fixedbb_multichain/lm_design_esm2_650m"


folder_with_pdbs="./inputs/PDB_complexes/pdbs/"

output_dir="./outputs/example_5_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi


path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
path_for_fixed_positions=$output_dir"/fixed_pdbs.jsonl"
path_for_tied_positions=$output_dir"/tied_pdbs.jsonl"
chains_to_design="A C"
fixed_positions="9 10 11 12 13 14 15 16 17 18 19 20 21 22 23, 10 11 18 19 20 22"
tied_positions="1 2 3 4 5 6 7 8, 1 2 3 4 5 6 7 8" #two list must match in length; residue 1 in chain A and C will be sampled togther;

python ./helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ./helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

python ./helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --chain_list "$chains_to_design" --position_list "$fixed_positions"

python ./helper_scripts/make_tied_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_tied_positions --chain_list "$chains_to_design" --position_list "$tied_positions"

rdebug ./design_pdb.py \
        --experiment_path $model_path --ckpt "best.ckpt" \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --fixed_positions_jsonl $path_for_fixed_positions \
        --tied_positions_jsonl $path_for_tied_positions \
        --out_dir $output_dir \
        --seed 42 \
        --num_seqs 2 \
        --temperature 1.0 \
        --max_iter 5