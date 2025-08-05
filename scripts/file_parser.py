import re
import os
import h5py
import numpy as np


def is_simulation_completed(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            upper_line = line.strip().upper()
            if "EXITED WITH ERRORS" in upper_line:
                return False
            if "COMPLETED" in upper_line:
                return True
    return False


def parse_inp_file(filepath):
    data = {"nodes": [], "elements": {}, "cloads": None}

    node_re = re.compile(r"^\*NODE\b", re.IGNORECASE)
    elem_re = re.compile(r"^\*ELEMENT\b", re.IGNORECASE)
    cload_re = re.compile(r"^\*CLOAD\b", re.IGNORECASE)
    section_start_re = re.compile(r"^\*")

    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**"):
                continue

            if node_re.match(line):
                current_section = "nodes"
                continue
            elif elem_re.match(line):
                current_section = "elements"
                continue
            elif cload_re.match(line):
                current_section = "cloads"
                continue
            elif section_start_re.match(line):
                current_section = None
                continue

            parts = [p.strip() for p in re.split(r",\s*", line)]
            if current_section == "nodes":
                try:
                    node_id = int(parts[0])
                    coords = list(map(float, parts[1:4]))
                    data["nodes"].append((node_id, *coords))
                except: pass
            elif current_section == "elements":
                try:
                    element_id = int(parts[0])
                    conn = tuple(map(int, parts[1:]))
                    data["elements"][element_id] = conn
                except: pass
            elif current_section == "cloads" and len(parts) == 3:
                try:
                    data["cloads"] = float(parts[2])
                except: pass

    return data


def extract_edges_from_elements(elements_dict):
    edges = set()
    for conn in elements_dict.values():
        conn = list(conn)
        for i in range(len(conn)):
            a, b = conn[i], conn[(i + 1) % len(conn)]
            edges.add((a, b))
            edges.add((b, a))
    return np.array(list(edges), dtype=np.int64).T


def extract_displacements(file_path):
    displacements = {}
    capture = False

    with open(file_path, 'r') as f:
        for line in f:
            if re.search(r'^\s*NODE\s+FOOT-', line):
                capture = True
                continue
            elif capture and line.strip().startswith("MAXIMUM"):
                break
            elif capture:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0])
                        values = list(map(float, parts[1:]))
                        displacements[node_id] = values
                    except:
                        continue
    return displacements


def save_to_hdf5(results, output_path="meshgraph_dataset_dgl.h5"):
    with h5py.File(output_path, "w") as h5f:
        for variant, data in results.items():
            nodes = data["inp_data"]["nodes"]
            elements = data["inp_data"]["elements"]
            displacements = data["displacements"]
            cload_val = data["inp_data"].get("cloads", 0.0)

            pos, spc, load, y = [], [], [], []
            node_ids = []

            for node_id, x, y_, z in nodes:
                u = displacements.get(node_id)
                if not u or len(u) < 3:
                    continue

                pos.append([x, y_, z])
                spc.append([1.0, 1.0, 1.0] if node_id in [1, 6, 9] else [0.0, 0.0, 0.0])
                load.append([0.0, 0.0, cload_val] if node_id == 2 else [0.0, 0.0, 0.0])
                y.append(u[:3])
                node_ids.append(node_id)

            if not pos:
                continue

            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            raw_edges = extract_edges_from_elements(elements)
            edge_list = [
                [node_id_to_idx[a], node_id_to_idx[b]]
                for a, b in raw_edges.T
                if a in node_id_to_idx and b in node_id_to_idx
            ]

            if not edge_list:
                continue

            edge_list = np.array(edge_list, dtype=np.int32)
            etypes = np.zeros(len(edge_list), dtype=np.float32)  # Dummy etype

            # ✅ Save as nested variant/case_000 structure
            group = h5f.create_group(f"{variant}/case_000")
            group.create_dataset("pos", data=np.array(pos, dtype=np.float32))
            group.create_dataset("spc", data=np.array(spc, dtype=np.float32))
            group.create_dataset("load", data=np.array(load, dtype=np.float32))
            group.create_dataset("y", data=np.array(y, dtype=np.float32))
            group.create_dataset("connectivity", data=edge_list)
            group.create_dataset("etypes", data=etypes)

    print(f"✅ DGL-compatible dataset saved at: {output_path}")




def print_variant_preview(variant, data):
    print(f"\n=== Preview: {variant} ===")
    print(f"Node Features (9D): [x, y, z, spc_x, spc_y, spc_z, force_x, force_y, force_z]")
    print(f"Edge Features (4D): [disp_diff_x, disp_diff_y, disp_diff_z]")
    print(f"Labels (3D): [disp_x, disp_y, disp_z]")
    print("-" * 120)

    nodes = data["inp_data"]["nodes"]
    displacements = data["displacements"]
    cload_val = data["inp_data"].get("cloads", 0.0)

    print(f"{'NodeID':>6} | {'Node Features (11D)':>60} | {'Labels (3D)':>30}")
    print("-" * 120)

    for i, (node_id, x, y, z) in enumerate(nodes[:9]):  # Show first 5 nodes
        disp = displacements.get(node_id)
        if disp and len(disp) >= 3:
            # Node features
            pos = [float(x), float(y), float(z)]
            spc = [1.0, 1.0, 1.0] if node_id in [1, 6, 9] else [0.0, 0.0, 0.0]
            force = [cload_val, 0.0, 0.0] if node_id == 2 else [0.0, 0.0, 0.0]
            node_feat = pos  + spc + force
            
            # Labels
            labels = disp[:3]
            
            node_feat_str = "[" + ", ".join(f"{f:6.3f}" for f in node_feat) + "]"
            labels_str = f"[{labels[0]:8.3e}, {labels[1]:8.3e}, {labels[2]:8.3e}]"
            
            print(f"{node_id:6d} | {node_feat_str:>60} | {labels_str:>30}")
    
    if len(nodes) > 9:
        print(f"... and {len(nodes) - 5} more nodes")



def main(base_folder="test_data", log_file_path="process_log.txt"):
    results = {}
    skipped = {}

    with open(log_file_path, "w", encoding="utf-8") as log:
        log.write("Processing Log\n====================\n")

        for folder in sorted(os.listdir(base_folder)):
            variant_path = os.path.join(base_folder, folder)
            if not os.path.isdir(variant_path): continue

            inp_file, dat_file, variant_log_file = None, None, None
            for fname in os.listdir(variant_path):
                fpath = os.path.join(variant_path, fname)
                if fname.lower().endswith(".inp"): inp_file = fpath
                elif fname.lower().endswith(".dat"): dat_file = fpath
                elif fname.lower().endswith(".log"): variant_log_file = fpath

            if not inp_file or not dat_file or not variant_log_file:
                skipped[folder] = "Missing required files"
                log.write(f"[SKIPPED] {folder}: Missing required files\n")
                continue

            if is_simulation_completed(dat_file):
                inp_data = parse_inp_file(inp_file)
                displacements = extract_displacements(dat_file)
                results[folder] = {
                    "inp_data": inp_data,
                    "displacements": displacements
                }
                log.write(f"[PROCESSED] {folder}: COMPLETED - Simulation completed\n")
            else:
                skipped[folder] = "Simulation incomplete"
                log.write(f"[SKIPPED] {folder}: FAILED - Simulation incomplete\n")

        if skipped:
            log.write("\nSummary of Skipped Folders:\n")
            for folder, reason in skipped.items():
                log.write(f"  - {folder}: {reason}\n")

    return results
   



if __name__ == "__main__":

  



    np.set_printoptions(precision=6, suppress=True)
    base_folder = os.path.join("..", "test_data")

    results = main(base_folder)
    save_to_hdf5(results)

    # print("\n📦 Saved variants in order (in meshgraph_dataset.h5):")
    # with h5py.File("meshgraph_dataset.h5", "r") as f:
    #     for name in f.keys():
    #         grp = f[name]
    #         node_feat_shape = grp["node_features"].shape
    #         edge_feat_shape = grp["edge_features"].shape
    #         labels_shape = grp["y"].shape
    #         print(f" - {name}: nodes={node_feat_shape}, edges={edge_feat_shape}, labels={labels_shape}")

    # Preview first variant
    for i, (variant, data) in enumerate(results.items()):
        print_variant_preview(variant, data)
        if i >= 0:  # Show first variant
            break
