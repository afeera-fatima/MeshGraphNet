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
                    coords = tuple(map(float, parts[1:4]))
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


def save_to_hdf5(results, output_path="meshgraph_dataset.h5"):
    with h5py.File(output_path, "w") as h5f:
        for variant, data in results.items():
            nodes = data["inp_data"]["nodes"]
            elements = data["inp_data"]["elements"]
            displacements = data["displacements"]
            cload_val = data["inp_data"].get("cloads", 0.0)

            node_features = []
            labels = []
            node_ids = []

            for node_id, x, y, z in nodes:
                disp = displacements.get(node_id)
                if disp and len(disp) >= 3:
                    try:
                        u = disp[:3]
                        cload = cload_val if node_id == 2 else 0.0
                        is_fixed = 1.0 if node_id in [1, 6, 9] else 0.0  # adjust as needed
                        node_features.append((x, y, z, cload, is_fixed))
                        labels.append(u)
                        node_ids.append(node_id)
                    except:
                        continue

            if not node_features:
                continue  # skip empty entries

            node_features = np.array(node_features)
            labels = np.array(labels)
            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            raw_edges = extract_edges_from_elements(elements)
            edge_index = [
                [node_id_to_idx[src], node_id_to_idx[tgt]]
                for src, tgt in raw_edges.T
                if src in node_id_to_idx and tgt in node_id_to_idx
            ]
            edge_index = np.array(edge_index) if edge_index else np.zeros((0, 2), dtype=int)
            etypes = np.zeros(edge_index.shape[0])  # dummy

            grp = h5f.create_group(variant)
            grp.create_dataset("pos", data=node_features[:, 0:3])
            grp.create_dataset("load", data=node_features[:, 3:4])
            grp.create_dataset("spc", data=node_features[:, 4:5])
            grp.create_dataset("thickness", data=np.ones((len(node_features), 1)))
            grp.create_dataset("ntypes", data=np.zeros((len(node_features), 1)))
            grp.create_dataset("y", data=labels)
            grp.create_dataset("connectivity", data=edge_index)
            grp.create_dataset("etypes", data=etypes)

    print(f"✅ Converted and saved dataset in ShellDataset format to: {output_path}")



def print_variant_preview(variant, data):
    print(f"\n=== Preview: {variant} ===")
    print(f"{'NodeID':>6} | {'x':>10} {'y':>10} {'z':>10} | {'cload':>10} {'is_fixed':>10} || {'U1':>10} {'U2':>10} {'U3':>10}")
    print("-" * 90)

    nodes = data["inp_data"]["nodes"]
    displacements = data["displacements"]
    cload_val = data["inp_data"].get("cloads", 0.0)

    for node_id, x, y, z in nodes:
        disp = displacements.get(node_id)
        if disp and len(disp) >= 3:
            u1, u2, u3 = disp[:3]
            cload = cload_val if node_id == 2 else 0.0
            is_fixed = 1.0 if node_id in [1, 6, 9] else 0.0
            print(f"{node_id:6d} | {x:10.4f} {y:10.4f} {z:10.4f} | {cload:10.2f} {is_fixed:10.0f} || {u1:10.6e} {u2:10.6e} {u3:10.6e}")



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
    #         print(f" - {name}")

    # # Preview first variant
    # for i, (variant, data) in enumerate(results.items()):
    #     print_variant_preview(variant, data)
    #     if i >= 1:
    #         break
