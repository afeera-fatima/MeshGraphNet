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
                    coords = list(map(float, parts[2:4])) #y and z coordinates
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
                        values = list(map(float, parts[2:4]))
                        displacements[node_id] = values
                    except:
                        continue
    return displacements

def save_to_hdf5(results, output_path="meshgraph_dataset_dgl.h5"):
    import os

    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"⚠️ Warning: Overwriting existing file at {output_path}")

    with h5py.File(output_path, "w") as h5f:
        for variant, data in results.items():
            try:
                nodes = data["inp_data"]["nodes"]
                elements = data["inp_data"]["elements"]
                displacements = data["displacements"]
                cload_val = data["inp_data"].get("cloads", 0.0)

                pos, spc, load, y = [], [], [], []
                node_ids = []

                for node_id, y_, z in nodes:
                    u = displacements.get(node_id, [0.0, 0.0])
                

                    pos.append([y_, z])
                    spc.append([1.0, 1.0] if node_id in [1, 6, 8] else [ 0.0, 0.0])
                    load.append([0.0, cload_val] if node_id == 2 else [ 0.0, 0.0])
                    y.append(u[:2])
                    node_ids.append(node_id)

                if not pos:
                    print(f"⚠️ Skipping variant {variant}: No valid nodes.")
                    continue

                node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                raw_edges = extract_edges_from_elements(elements)
                edge_list = [
                    [node_id_to_idx[a], node_id_to_idx[b]]
                    for a, b in raw_edges.T
                    if a in node_id_to_idx and b in node_id_to_idx
                ]

                if not edge_list:
                    print(f"⚠️ Skipping variant {variant}: No valid edges.")
                    continue

                edge_list = np.array(edge_list, dtype=np.int32)
               
                # ✅ Save as nested variant/case_000 structure
                group = h5f.create_group(f"{variant}/case_000")
                group.create_dataset("pos", data=np.array(pos, dtype=np.float32))
                group.create_dataset("spc", data=np.array(spc, dtype=np.float32))
                group.create_dataset("load", data=np.array(load, dtype=np.float32))
                group.create_dataset("y", data=np.array(y, dtype=np.float32))
                group.create_dataset("connectivity", data=edge_list)
        

                print(f"✅ Saved variant: {variant}")

            except Exception as e:
                print(f"❌ Error saving variant {variant}: {e}")

    print(f"✅ DGL-compatible dataset saved at: {output_path}")


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

   