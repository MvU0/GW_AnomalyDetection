# Configuration of training for models discussed in thesis: 
            Learning Auxiliary Channel Relationships for Glitch Detection in Gravitational Wave Detectors Using Graph Neural Networks


# Time series:
The same across both models:
n_epochs = 500
T_max = 500
batch_size = 32
train_frac = 0.7
val_frac = 0.15
num_layers = 1
mode = 'mean'
balance_classes = True
seed = 15


-- Channels as nodes:
dropout = 0.055
topk = 10
patience = 108
pool_type = 'mean'
similarity_type = 'cosine
eta_min = 5e-8
lr = 1e-4

-- Time steps as nodes:
dropout = 0.00037
topk = 31
patience = 99
pool_type = 'max'
similarity_type = 'cosine
eta_min = 1e-9
lr = 5e-4
                


# Fractal dimension:
The same across both models:
n_epochs = 750
T_max = 750
seed = 15
batch_size = 32
train_frac = 0.7
val_frac = 0.15
num_layers = 1
balance_classes=True

-- Channels as nodes
dropout = 0.562
topk = 46
patience = 142
pool_type = 'max'
similarity_type = 'pearson'
mode = 'concat'
preprocess = True
eta_min = 1e-7
lr = 5e-4



-- Time dimension as nodes
dropout = 0.518
topk = 10
patience = 65
pool_type = 'mean
similarity_type = 'pearson'
mode = 'mean'
preprocess = True
eta_min = 5e-9
lr = 5e-5