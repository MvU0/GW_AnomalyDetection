import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.data import save_scores_to_hdf5, get_subsys_masks, generate_datasets

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores, process_labels


import time
from datetime import datetime

import os
import argparse
from pathlib import Path

import json
import random



class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None


        self.sample_len = (2048-self.train_config['slide_win']) # Save length used per sample for ease later in the code
        self.slide_stride = self.train_config['slide_stride']
        self.details = train_config['comment']

        print(f"Window: {train_config['slide_win']}")
        print(f"Topk: {train_config['topk']}")

        dataset = self.env_config['dataset']
        
        # Load data
        normal_data = pd.read_parquet(f'./data/{dataset}/normal_data.parquet')
        anomaly_data = pd.read_parquet(f'./data/{dataset}/anomaly_data_extended.parquet')
        
        # Generate boolean masks for subsystems + create dictionary mapping name of channel to a subsystem -> used to colour the graph in the end
        subsys_masks, channel_to_subsystem_full = get_subsys_masks(channel_names = list(normal_data.keys()))
        self.subsys_masks = subsys_masks
        self.channel_to_subsystem = channel_to_subsystem_full
        
        if self.env_config['subsystems'] == 'FULL':
            combined_mask = np.ones(normal_data.shape[1], dtype=bool)
            subsystem_names = ['FULL']

        else:
            # Parse subsytem list
            subsystem_names = [s.strip() for s in self.env_config['subsystems'].split(',')]
            # Validate subsystem names
            for subsystem in subsystem_names:
                if subsystem not in subsys_masks:
                    raise ValueError(f"Unknown subssytem: {subsystem}")
            # Combine masks
            combined_mask = np.zeros_like(subsys_masks[subsystem_names[0]], dtype=bool)
            for subsystem in subsystem_names:
                combined_mask += subsys_masks[subsystem]

            

        print(f"Using subsystems: {subsystem_names}")

        # Split data in train/val/test and applying mask
        train, val, test = generate_datasets(normal_data, anomaly_data, mask = combined_mask, val_percentage=train_config['val_ratio'], test_percentage=train_config['test_ratio'])
        print(f"Train data shape: {train.shape}", flush=True)
        print(f"Val data shape: {val.shape}", flush=True)
        print(f"Test data shape: {test.shape}", flush=True)
        print(train.memory_usage(deep=True).sum() / 1e9, "GB train", flush=True)
        print(val.memory_usage(deep=True).sum() / 1e9, "GB val", flush=True)
        print(test.memory_usage(deep=True).sum() / 1e9, "GB test", flush=True)
                
       
        feature_map = get_feature_map(train)
        
        fc_struc = get_fc_graph_struc(dataset)
        
        set_device(env_config['device'])
        self.device = get_device()
        
        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.fc_edge_index = fc_edge_index
        
        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        val_dataset_indata = construct_data(val, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist()) 

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg, segment_length=2048) # Segments are 1648 timesteps (2048 (8secs) - 400 (200 on both sides to get rid of whitening edge effects))
        val_dataset = TimeDataset(val_dataset_indata, fc_edge_index, mode='test', config=cfg, segment_length=2048)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg, segment_length=2048)   # However not necesarily whitened data so then it would be 2048 steps

        # Put data in DataLoader
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'],
                                shuffle=True, pin_memory=True, num_workers = 2)

        self.val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch'],
                                shuffle=False, pin_memory=True, num_workers = 2)
        
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, pin_memory=True, num_workers=2)          # num_workers > 0 only matters if there needs to be some calculations done to get the item
                                                                                    # num_workers paralelises gathering the data, but if the data is easily gatherable it doesn't change anything
                                                                                    # And generating workers takes a while 

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        
        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)
        print(f"Created model on {self.device} device")

        

    def run(self):
        
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
            self.run_dir = Path(model_save_path).parent
        else:
            model_save_path, results_csv_path = self.get_save_path()
            self.run_dir = Path(results_csv_path).parent

        self.save_config()

        self.train_loss, self.val_loss, self.lr_list = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
            )
        
        self.save_plots()
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result, values = test(best_model, self.test_dataloader)
        _, self.val_result, _ = test(best_model, self.val_dataloader)

        torch.save(values, f"./_analysis/values/{self.env_config['dataset']}{self.details}.pt") # This saves the predicted values and actual values
        # Timesteps per sample is 2048 (8s at sampling rate 256) - sliding window size (train_config['slide_win'])
          

        #self.get_score(self.test_result, self.val_result, run_mean_window=10, sample_len=self.sample_len)
        self.get_score(self.test_result, self.val_result, run_mean_window=50, sample_len=self.sample_len)
        self.get_score(self.test_result, self.val_result, run_mean_window=100, sample_len=self.sample_len)
        #self.get_score(self.test_result, self.val_result, run_mean_window=200, sample_len=self.sample_len)
        self.get_score(self.test_result, self.val_result, run_mean_window=500, sample_len=self.sample_len)



    def get_score(self, test_result, val_result, run_mean_window, sample_len):
        _start = time.time()

        np_test_result = np.array(test_result)

        test_labs = np_test_result[2, :, 0].tolist()
        test_labels = process_labels(test_labs, sample_len, run_mean_window)
        print("Test labels shape:", np.array(test_labels).shape)
        test_scores, normal_scores = get_full_err_scores(test_result, val_result, run_mean_window=run_mean_window, sample_len=sample_len)

        
        print('=========================** Result **============================\n', flush=True)
        info = None
        if self.env_config['report'] == 'best':
            top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)
            info = top1_val_info

        f1, precision, recall, auc_score, threshold, pred_labels, gt_labels, total_topk_err_scores, topk_indices = info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7], info[8]
        self.topk_indices = topk_indices

        print(f'Classification window size: {run_mean_window}', flush=True)
        print(f'F1 score: {f1}', flush=True)
        print(f'precision: {precision}', flush=True)
        print(f'recall: {recall}\n', flush=True)

        # Save results to CSV here
        save_path = self.get_save_path()[1]  # second item: results CSV path
        results_df = pd.DataFrame([{
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC_score': auc_score,
            'Threshold': threshold,
            'Dataset': self.env_config['dataset'],
            'Report': self.env_config['report'],
            'Timestamp': self.datestr
        }])

        p = Path(save_path)
        results_path = p.with_suffix('') / f"{run_mean_window}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

        # Saving test_scores in HDF5 format! (save_scores_to_hdf5 comes from util/data.py)
                                                                   
        save_scores_to_hdf5(
            scores=test_scores,
            base_path=f"./_analysis/err_scores/test_scores{self.details}",
            segment_starts=None,                        # Can also set to None if you don't want data saved by segment
        )
        # normal_scores
        normal_scores_df = pd.DataFrame(normal_scores)
        normal_scores_df.to_csv(f"./_analysis/err_scores/normal_scores{self.details}.csv", index=False)

        
        labels_df = pd.DataFrame({
            'Predicted labels': pred_labels,
            'Ground truth labels': gt_labels,
            'total_topk_err_scores' : total_topk_err_scores,
            'top_error_index' : self.topk_indices[0,:]
        })

        dir_path = Path(f"./_analysis/labels/{self.env_config['dataset']}/{self.details}")
        dir_path.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(dir_path / f"{run_mean_window}.csv", index = False)

        _end = time.time()
        print(f'Time for saving score with window size {run_mean_window}: {_end - _start} s', flush=True)

    def get_save_path(self):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}_{self.details}.pt',
            f'./results/{dir_path}/{datestr}_{self.details}.csv', # Check if this needs changing after changing how I save my results in get_score
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

    def save_plots(self):
        # Plot the training and validation loss
        save_dir = Path(
            f"./results/{self.env_config['dataset']}/"
            f"{self.datestr}_{self.details}"
        )
        out_dir = save_dir / "imgs"

        print(f"Loss plots saved in: {out_dir}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        epochs = np.arange(1, len(self.train_loss) + 1)
        plt.figure(figsize=(8,5))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(self.train_loss, label='Train Loss')
        ax1.plot(self.val_loss, label='Validation Loss')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.lr_list, linestyle='--')
        ax2.set_ylabel('Learning rate')
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig(out_dir / f"loss_curve_{self.details}.png")
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / f"loss_curve_log_{self.details}.png")
        plt.close()


        # Retrieve embedding
        embedding_test = self.model.embedding.weight.detach().cpu().numpy()
        channel_names = self.feature_map
        
        assert len(channel_names) == embedding_test.shape[0]
        
        # Do t-SNE
        tsne = TSNE(n_components=2, perplexity=25, random_state=42)
        emb_2d = tsne.fit_transform(embedding_test)
        
        # Subsystem mapping + plotting setup
        channel_to_subsystem = self.channel_to_subsystem
        subsystems = sorted(set(channel_to_subsystem.values()))
        # Set colormap logic
        cmap = plt.get_cmap("tab10")


        def mpl_to_plotly_color(c):
            """Convert matplotlib RGBA (0-1) to plotly rgba string"""
            r, g, b, a = c
            return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"

        subsystem_colors = {
            subsys: mpl_to_plotly_color(cmap(i % 10))
            for i, subsys in enumerate(subsystems)
        }

        # Build dataframe for Plotly
        df = pd.DataFrame({
            "x": emb_2d[:, 0],
            "y": emb_2d[:, 1],
            "detector": channel_names,
            "subsystem": [channel_to_subsystem.get(ch, "UNKNOWN") for ch in channel_names]
        })

        # Create interactive plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="subsystem",
            color_discrete_map=subsystem_colors,
            hover_name="detector",  # shows full detector name on hover
            title="Node embeddings colored by subsystem",
            render_mode="webgl"  # better for large datasets
        )

        # Styling (similar to your matplotlib settings)
        fig.update_traces(marker=dict(size=8, opacity=0.8))

        fig.update_layout(
            width=1000,
            height=800,
            legend_title_text="Subsystem"
        )

        # Save interactive HTML
        emb_dir = Path(f"./embeddings/{self.env_config['dataset']}")
        emb_dir.mkdir(parents=True, exist_ok=True)

        fig.write_html(emb_dir / f"{self.details}.html")

    def save_config(self):

        config_dict = {
            'train_config': self.train_config,
            'env_config': self.env_config
        }

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')

        save_dir = Path(
            f"./results/{self.env_config['dataset']}/"
            f"{self.datestr}_{self.details}"
        )

        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.json"

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        print(f"Saved config to: {config_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='GW', type = str, default='GW')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-test_ratio', help='test ratio', type = float, default=0.1)
    parser.add_argument('-learning_rate', help='learning rate', type = float, default=1e-3)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-subsystems', help='Comma-separated subsystem names', type=str, default='SUS')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'learning_rate': args.learning_rate,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'subsystems': args.subsystems
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()
    

