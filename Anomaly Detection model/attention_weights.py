import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets.TimeDataset import TimeDataset
from util.preprocess import construct_data
import os, sys


def attention_weights(model, feature_map, fc_edge_index, dataset_name, topk_indices, device, slide_win=15, slide_stride=5,batch_size=1):
    save_index = [14863,14942]  # Start and stop point of index saving
    """
    Extract and save attention weights from a trained GDN model
    using the same preprocessing and dataset structure as training.
    """
    #print(topk_indices)
    #print(topk_indices.shape)
    print("\n================ Extracting Attention Weights ================")

    # --- Load test data ---
    test_path = f'./data/{dataset_name}/test.csv'
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    test_df = pd.read_csv(test_path, index_col=0)
    if 'attack' in test_df.columns:
        test_df = test_df.drop(columns=['attack'])
    
    # --- Rebuild edge index and input data like during training ---
    test_dataset = construct_data(test_df, feature_map, labels=0)

    # --- Load edge index and dataset ---
    cfg = {'slide_win': slide_win, 'slide_stride': slide_stride}

    test_dataset = TimeDataset(test_dataset, fc_edge_index, mode='test', config=cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- Prepare model ---
    model.eval()
    model = model.to(device)

    all_attention_data = []

    # --- Go through one batch (or all, if you want) ---
    with torch.no_grad():
        for i, (x, y, label, edge_index) in enumerate(test_loader):
            x, edge_index = x.to(device).float(), edge_index.to(device)
            if i not in range(save_index[0], save_index[1]):  # only extract attention for the first batch
                continue

            x, edge_index = x.to(device), edge_index.to(device)
            _ = model(x, edge_index)  # forward pass triggers attention storage

            top_index = topk_indices[0,i]
            layer_attention = []
            for layer_id, layer in enumerate(model.gnn_layers):
                if hasattr(layer, "att_weight_1"):
                    att = layer.att_weight_1.detach().cpu()
                    edges = layer.edge_index_1.detach().cpu()
                    layer_attention.append({
                        "layer": layer_id,
                        "edge_index": edges,
                        "attention_weights": att,
                        "top_index": top_index
                    })

            all_attention_data.append({
                "sample_id": i,
                "layers": layer_attention
            })
    
    # --- Save results ---
    os.makedirs("./_analysis", exist_ok=True)
    save_path = f"./_analysis/attention_weights/glitch_10.pt"
    torch.save(all_attention_data, save_path)
    print(f" Attention weights saved to: {save_path}")

    return all_attention_data
