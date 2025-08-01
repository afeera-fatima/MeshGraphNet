""""""

import os
import re
import torch
import numpy as np
from typing import Any, List, Union
import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Dataset requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
try:
    import vtk
except ImportError:
    raise ImportError(
        "Dataset requires the vtk and pyvista libraries. Install with "
        + "pip install vtk pyvista"
    )
import h5py

class Hdf5Dataset:
    """
    split_arr: array containing shuffled inidices tuples of data (separate for train,val etc)
    data_path: path to
    """

    def __init__(self, data_path, split_arr, length):
        self.data_path = data_path
        self.split_arr = split_arr
        self.length = length

    def _get_file(self):
        if not hasattr(self, "_data_file"):
            self._data_file = h5py.File(self.data_path, "r")
        return self._data_file

    def __getitem__(self, id):
        data_file = self._get_file()
        variant, subcase = self.split_arr[id]
        group = data_file[variant][subcase]
        # Convert the h5py.Group into a dictionary
        data_point = {key: group[key][()] for key in group.keys()}
        return data_point

    def __len__(self):
        return self.length


class ShellDataset(DGLDataset):
    def __init__(
        self,
        split,
        dataset_split,
        num_samples=10,
        invar_keys=["pos", "ntypes", "thickness", "spc", "load"],
        outvar_keys=["disp_x", "disp_y", "disp_z"],
        normalize_keys=["disp_x", "disp_y", "disp_z", "load", "thickness"],
        force_reload=False,
        name="dataset",
        verbose=False,
        normalization="z_score",
    ):
        """
        dataset_split: Hdf5Dataset object
        split: name of split
        """
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.etypes = []
        self.num_samples = num_samples
        self.input_keys = invar_keys
        self.output_keys = outvar_keys
        self.dataset_split = dataset_split
        self.normalize_keys = normalize_keys

        print(f"Preparing the {split} dataset...")
        self.length = min(len(self.dataset_split), self.num_samples)

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = []
        for i in range(self.length):
            data_i = self.dataset_split[i]
            graph = self._create_dgl_graph(data_i)
            self.graphs.append(graph)

        if normalization == "z_score":
            if split == "train":
                self.node_stats = self._get_node_stats_for_zscore(keys=self.normalize_keys)
                self.edge_stats = self._get_edge_stats_for_zscore()
            else:
                self.node_stats = load_json("node_stats.json")
                self.edge_stats = load_json("edge_stats.json")

            # labels are added here
            self.graphs = self.z_score_norm_node()
            self.graphs = self.z_score_norm_edge()

        elif normalization == "max_abs":
            self.graphs = self.max_abs_norm()

        elif normalization == "custom":
            if split == "train":
                self.edge_stats = self._get_edge_stats_for_zscore()
                self.node_stats = self._get_node_stats_for_robust_scaler(normalize_keys[:3]) # displaccements here
                self.node_stats |= self._get_node_stats_minmax(normalize_keys[3:]) # thickness and load here
            else:
                self.edge_stats = load_json("edge_stats.json")
                self.node_stats = load_json("node_stats_robust_scaler.json")
                self.node_stats |= load_json("node_stats_minmax.json") # works like .update() for dicts
            
            self.graphs = self.min_max_norm_node()
            self.graphs = self.robust_scaler_node_disp()
            self.graphs = self.z_score_norm_edge()

        elif normalization == "min-max":
            if split == "train":
                self.edge_stats = self._get_edge_stats_for_zscore()
                self.node_stats = self._get_node_stats_minmax(normalize_keys)
            else:
                self.edge_stats = load_json("edge_stats.json")
                self.node_stats = load_json("node_stats_minmax.json")

            self.graphs = self.min_max_norm_node()
            self.graphs = self.z_score_norm_edge()

        # Concat normalized features
        self._concat_features()

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

    def __len__(self):
        return self.length

    def _concat_features(self):
        for i in range(len(self.graphs)):
            # Concatenate input features into "x"
            self.graphs[i].ndata["x"] = torch.cat(
                [
                    (
                        self.graphs[i].ndata[key].view(-1, 1)
                        if self.graphs[i].ndata[key].dim() == 1
                        else self.graphs[i].ndata[key]
                    )
                    for key in self.input_keys
                ],
                dim=-1,
            )

            # Concatenate output features into "y"
            self.graphs[i].ndata["y"] = torch.cat(
                [
                    (
                        self.graphs[i].ndata[key].view(-1, 1)
                        if self.graphs[i].ndata[key].dim() == 1
                        else self.graphs[i].ndata[key]
                    )
                    for key in self.output_keys
                ],
                dim=-1,
            )

            # Concat edge features
            self.graphs[i].edata["x"] = torch.cat(
                (self.graphs[i].edata["x"], self.etypes[i].view(-1, 1)), dim=-1
            )

    def max_abs_norm(self):
        for i in range(len(self.graphs)):
            self.graphs[i].ndata["max_abs"] = {}
            for key in self.normalize_keys:
                # Apply Max-Abs normalization to each node feature
                max_abs = torch.max(torch.abs(self.graphs[i].ndata[key]))
                self.graphs[i].ndata["max_abs"][key] = max_abs
                self.graphs[i].ndata[key] = self.graphs[i].ndata[key] / max_abs

            # Apply Max-Abs normalization to edge features
            max_abs_edge = torch.max(torch.abs(self.graphs[i].edata["x"]))
            self.graphs[i].edata["x"] = self.graphs[i].edata["x"] / max_abs_edge

        return self.graphs

    def max_abs_denorm(self, data, max_abs):
        return data * max_abs

    def z_score_norm_node(self):
        """normalizes node features"""
        invar_keys = set(
            [
                key.replace("_mean", "").replace("_std", "")
                for key in self.node_stats.keys()
            ]
        )
        epsilon = torch.tensor(1e-8, dtype=torch.float32)

        for key in invar_keys:
            for i in range(len(self.graphs)):
                self.graphs[i].ndata[key] = (
                    self.graphs[i].ndata[key] - self.node_stats[key + "_mean"]
                ) / (self.node_stats[key + "_std"] + epsilon)

        return self.graphs

    def z_score_norm_edge(self):
        """normalizes a tensor"""
        for i in range(len(self.graphs)):
            self.graphs[i].edata["x"] = (
                self.graphs[i].edata["x"] - self.edge_stats["edge_mean"]
            ) / self.edge_stats["edge_std"]

        return self.graphs

    @staticmethod
    def z_score_denorm(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar
    
    def _get_node_stats_minmax(self, keys):
        stats = {}

        for key in keys:
            stats[key + "_min"] = 0
            stats[key + "_max"] = 0
            node_values = []

            for i in range(self.length):
                node_values.append(self.graphs[i].ndata[key])

            node_tensor = torch.cat(node_values, dim=0)
            stats[key + "_min"] = torch.min(node_tensor, dim=0).values
            stats[key + "_max"] = torch.max(node_tensor, dim=0).values

        # save to file
        save_json(stats, "node_stats_minmax.json")
        return stats
    
    def min_max_norm_node(self, feature_range=(0,1), epsilon=1e-8):
        """Applies Min-Max Normalization to node features"""
        a, b = feature_range
        for key in self.normalize_keys:
            min_val = self.node_stats[key + "_min"]
            max_val = self.node_stats[key + "_max"]

            for i in range(len(self.graphs)):
                self.graphs[i].ndata[key] = a + \
                ((self.graphs[i].ndata[key] - min_val)  / (max_val - min_val + epsilon)) \
                * (b - a)

        return self.graphs

    @staticmethod
    def min_max_denorm(data_arr, old_min,  old_max, feature_range=(0,1), epsilon=1e-8):
        new_min,new_max = feature_range
        denormalized = old_min + (old_max - old_min) * ((data_arr - new_min) / (new_max - new_min + epsilon))
        return denormalized

    def _get_node_stats_for_robust_scaler(self, keys):
        stats = {}

        for key in keys:
            stats[key + "_median"] = 0
            stats[key + "_iqr"] = 0
            node_values = []

            for i in range(self.length):
                node_values.append(self.graphs[i].ndata[key])

            node_tensor = torch.cat(node_values, dim=0)
            stats[key + "_median"] = torch.median(node_tensor, dim=0).values
            q1 = torch.quantile(node_tensor, 0.25, dim=0)
            q3 = torch.quantile(node_tensor, 0.75, dim=0)
            stats[key + "_iqr"] = q3 - q1

        # save to file
        save_json(stats, "node_stats_robust_scaler.json")
        return stats
    
    def robust_scaler_node_disp(self):
        """Applies RobustScaler to node features"""
        invar_keys = set(
            [
                key.replace("_median", "").replace("_iqr", "")
                for key in self.node_stats.keys()
                if key.endswith('_median') or key.endswith('_iqr')
            ]
        )

        for key in invar_keys:
            for i in range(len(self.graphs)):
                self.graphs[i].ndata[key] = (
                    self.graphs[i].ndata[key] - self.node_stats[key + "_median"]
                ) / (self.node_stats[key + "_iqr"])

        return self.graphs

    @staticmethod
    def robust_scaler_denorm(invar, median, iqr):
        """Denormalizes a tensor using RobustScaler"""
        denormalized_invar = invar * iqr + median
        return denormalized_invar

    def _get_edge_stats_for_zscore(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.length):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.length
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0) / self.length
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats_for_zscore(self, keys):
        stats = {}
        for key in keys:
            stats[key + "_mean"] = 0
            stats[key + "_meansqr"] = 0

        for i in range(self.length):
            for key in keys:
                stats[key + "_mean"] += (
                    torch.mean(self.graphs[i].ndata[key], dim=0) / self.length
                )
                stats[key + "_meansqr"] += (
                    torch.mean(torch.square(self.graphs[i].ndata[key]), dim=0)
                    / self.length
                )

        for key in keys:
            stats[key + "_std"] = torch.sqrt(
                stats[key + "_meansqr"] - torch.square(stats[key + "_mean"])
            )
            stats.pop(key + "_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    def _create_dgl_graph(self, data_i, to_bidirected=True, dtype=torch.int32):
        edge_list = data_i["connectivity"].tolist()
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)

        graph.ndata["pos"] = torch.tensor(data_i["pos"], dtype=torch.float32)
        graph.ndata["ntypes"] = torch.tensor(data_i["ntypes"], dtype=torch.float32)
        graph.ndata["thickness"] = torch.tensor(
            data_i["thickness"], dtype=torch.float32
        )
        graph.ndata["spc"] = torch.tensor(data_i["spc"], dtype=torch.float32)
        graph.ndata["load"] = torch.tensor(data_i["load"], dtype=torch.float32)

        for i, arr_name in enumerate(self.output_keys):
            graph.ndata[arr_name] = torch.tensor(data_i["y"][:, i], dtype=torch.float32)

        pos = graph.ndata["pos"]
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

        # Adjust etypes for bidirected graph
        etypes = torch.tensor(data_i["etypes"], dtype=torch.float32)
        if to_bidirected:
            etypes = torch.cat(
                [etypes, etypes]
            )  # Duplicate etypes for bidirected edges

        self.etypes.append(etypes)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return graph
