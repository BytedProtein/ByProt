mkdir -p data
wget -r -nd -np https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz -P data/multichain
cd data/multichain
tar -xzvf pdb_2021aug02_sample.tar.gz 