# ipm-acc-gnn
Code for generating training and test data, model training, and evaluation. Used in my master thesis: Accelerating Interior Point Methods using Graph Neural Networks.

# Install dependencies
pip install -r requirements.txt

# Usage: Generating training data, training model, and testing model
python3 experimentPipeline.py \
python3 precondModel.py \ 
python3 ipm_torch.py  \

Each of the above scripts takes a keyword: --size. This is either small (default) or large for experimentPipeline and precondModel, were ipm_torch also accepts unseen. 

