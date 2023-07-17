pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .
pip install -e vendor/esm

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch_geometric biotite
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
