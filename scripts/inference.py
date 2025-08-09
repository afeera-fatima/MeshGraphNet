# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import torch
from hydra.utils import to_absolute_path
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.utils import load_checkpoint
from meshgraphnet import MeshGraphNet
from omegaconf import DictConfig

from utils import mse, load_test_idx, create_vtk_from_graph

try:
    from dgl.dataloading import GraphDataLoader
except Exception:
    raise ImportError(
        "Stokes  example requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    pass
except Exception:
    raise ImportError(
        "Stokes  Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )
from datapipe.shell_dataset import Hdf5Dataset, ShellDataset


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger):
        self.logger = logger
        self.results_dir = cfg.results_dir

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        test_idx = load_test_idx("/home/sces201/Afeera/ML_Task/scripts/shell_mgn/low_precision_outputs/test_idx.pt")

        test_hdf5 = Hdf5Dataset(cfg.data_path, test_idx, len(test_idx))
        self.dataset = ShellDataset(
            name="shell_test",
            dataset_split=test_hdf5,
            split="test",
            num_samples=cfg.num_test_samples,
            normalization=cfg.normalization,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        )
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
            # epoch=48,  #### change to load ckpt of choice, or None for loading latest saved
        )

    # def predict(self):
    #     """
    #     Run the prediction process.

    #     Parameters:
    #     -----------
    #     save_results: bool
    #         Whether to save the results in form of a .vtp file, by default False

    #     Returns:
    #     --------
    #     None
    #     """

    #     self.pred, self.graphs = [], []
    #     stats = {
    #         key: value.to(self.device) for key, value in self.dataset.node_stats.items()
    #     }
    #     for i, graph in enumerate(self.dataloader):
    #         graph = graph.to(self.device)
    #         pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()
    #         pred = torch.cat([torch.zeros((pred.shape[0], 1), device=self.device), pred], dim=1)

    #         keys = ["disp_x", "disp_y", "disp_z"]
    #         ### read graph_data/ create polydata
    #         data_i = self.dataset.dataset_split[i]
    #         polydata = create_vtk_from_graph(data_i)
    #         graph.ndata["y"] = torch.cat([graph.ndata["y"], torch.zeros((pred.shape[0], 1), device=self.device)], dim=1)
    #         y_val = graph.ndata["y"].detach().cpu().numpy()
    #         with torch.no_grad():
    #             for key_index, key in enumerate(keys):
    #                 pred_val = pred[:, key_index : key_index + 1]
    #                 target_val = graph.ndata["y"][:, key_index : key_index + 1]

    #                 if key == "disp_x":
    #                     continue
    #                 pred_val = self.dataset.z_score_denorm(
    #                     pred_val, stats[f"{key}_mean"], stats[f"{key}_std"]
    #                 )
    #                 target_val = self.dataset.z_score_denorm(
    #                     target_val, stats[f"{key}_mean"], stats[f"{key}_std"]
    #                 )

    #                 error = mse(pred_val, target_val)
    #                 self.logger.info(
    #                     f"Sample {i} - mse error of {key} (%): {error:.3f}"
    #                 )

    #                 polydata[f"pred_{key}"] = pred_val.detach().cpu().numpy()

    #         print(polydata["pred_disp_y"], polydata["pred_disp_z"])
    #         self.logger.info("-" * 50)
    #         os.makedirs(to_absolute_path(self.results_dir), exist_ok=True)
    #         polydata.save(
    #             os.path.join(to_absolute_path(self.results_dir), f"shell_graph_{i}.vtp")
    #         )


#for ground truth

    def predict(self):
        # """
        # Run the prediction process.
        # """
        self.pred, self.graphs = [], []
        stats = {key: value.to(self.device) for key, value in self.dataset.node_stats.items()}

        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()
            pred = torch.cat([torch.zeros((pred.shape[0], 1), device=self.device), pred], dim=1)

            keys = ["disp_x", "disp_y", "disp_z"]

            data_i = self.dataset.dataset_split[i]
            polydata = create_vtk_from_graph(data_i)
            
            graph.ndata["y"] = torch.cat(
                [graph.ndata["y"], torch.zeros((pred.shape[0], 1), device=self.device)],
                dim=1
            )
            coordinates = torch.tensor(data_i["pos"], dtype=torch.float32, device=self.device)
            coordinates = torch.cat(
                [torch.zeros((pred.shape[0], 1), device=self.device), coordinates],
                dim=1
            )
            denorm_preds = torch.zeros((3, 9), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for key_index, key in enumerate(keys):
                    pred_val = pred[:, key_index : key_index + 1]
                    target_val = graph.ndata["y"][:, key_index : key_index + 1]

                    if key == "disp_x":
                        continue
                    # Denormalize
                    pred_val = self.dataset.z_score_denorm(
                        pred_val, stats[f"{key}_mean"], stats[f"{key}_std"]
                    )
                    target_val = self.dataset.z_score_denorm(
                        target_val, stats[f"{key}_mean"], stats[f"{key}_std"]
                    )
                    # Calculate MSE
                    error = mse(pred_val, target_val)
                    self.logger.info(f"Sample {i} - mse error of {key} (%): {error:.3f}")

                    # Save both prediction & ground truth into polydata
                    polydata[f"pred_{key}"] = pred_val.detach().cpu().numpy()
                    polydata[f"gt_{key}"] = target_val.detach().cpu().numpy()
                    # coordinates[:, key_index-1] = pred_val.squeeze()
                    # Print ALL values
                    print(f"\nSample {i} - {key} predictions vs ground truth:")
                    print("Pred:", pred_val.cpu().numpy().flatten())
                    print("GT  :", target_val.cpu().numpy().flatten())
                    
                     # Store predicted value into the coordinates array
                    denorm_preds[key_index] = pred_val.squeeze()    
                            
            # Convert final coordinates tensor to numpy
            coordinates = coordinates.cpu().numpy()
            denorm_preds = denorm_preds.cpu().numpy().T
            displaced_coords = coordinates + denorm_preds
            
            self.logger.info("-" * 50)
            # Save .vtp file with predictions + ground truth
            # save .inp file 
        
            self.write_inp_file(displaced_coords,coordinates ,to_absolute_path(f"/home/sces201/Afeera/ML_Task/scripts/shell_mgn/low_precision_results_inp_displaced_coords_{i}.inp"))


            # os.makedirs(to_absolute_path(self.results_dir), exist_ok=True)
            # polydata.save(
            #     os.path.join(to_absolute_path(self.results_dir), f"shell_graph_{i}.vtp")
            # )

    def write_inp_file(self, displaced_coords,coordinates, filename):
        """
        Write the displaced coordinates to a .inp file.

        Parameters:
        -----------
        displaced_coords: numpy.ndarray
            The displaced coordinates to write to the file.
        filename: str
            The name of the output .inp file.
        """
        with open(filename, 'w') as f:
            f.write("*Node\n")
            for i, coord in enumerate(displaced_coords):
                f.write(f"{i + 1}, {coord[0]}, {coord[1]}, {coord[2]}\n")
            f.write("*End Node\n")

            f.write("*\n\n\n\nGROUND TRUTH\n")
            for i, coord in enumerate(coordinates):
                f.write(f"{i + 1}, {coord[0]}, {coord[1]}, {coord[2]}\n")
            
        self.logger.info(f"Displaced coordinates saved to {filename}")
        

@hydra.main(version_base="1.3", config_path="conf/single_run_conf", config_name="inference_conf")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = MGNRollout(cfg, logger)
    rollout.predict()


if __name__ == "__main__":
    main()