# Abaqus/Analysis exited with errors
# Abaqus JOB test_1 COMPLETED
import re
import os
import h5py
import numpy as np

def is_simulation_completed(filepath):
    success_keywords = [
        "COMPLETED",
    ]
    error_keywords = [
        "EXITED WITH ERRORS",  
    ]

    with open(filepath, 'r') as file:
        for line in file:
            upper_line = line.strip().upper()
            if any(error in upper_line for error in error_keywords):
                return False
            if any(success in upper_line for success in success_keywords):
                return True

    return False  # Default to False if success not confirmed


def parse_inp_file_with_re(filepath):
    data = {
        "nodes": [],
        "elements": {},
        "cloads": None,
    }

    # Compile regex for section headers (case-insensitive)
    node_section_re = re.compile(r"^\*NODE\b", re.IGNORECASE)
    element_section_re = re.compile(r"^\*ELEMENT\b", re.IGNORECASE)
    cload_section_re = re.compile(r"^\*CLOAD\b", re.IGNORECASE)
    section_start_re = re.compile(r"^\*")  # any section start

    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**"):
                continue

            # Detect section headers with regex
            if node_section_re.match(line):
                current_section = "nodes"
                continue
            elif element_section_re.match(line):
                current_section = "elements"
                continue
            elif cload_section_re.match(line):
                current_section = "cloads"
                continue
            elif section_start_re.match(line):
                current_section = None
                continue

            if current_section == "nodes":
                # Expecting lines like: 1, -94.20163726309161, 3.030303, 5.652396
                # Use regex to split on commas and strip spaces
                parts = [p.strip() for p in re.split(r",\s*", line)]
                try:
                    node_id = int(parts[0])
                    coords = tuple(float(x) for x in parts[1:4])
                    data["nodes"].append((node_id, *coords))
                except (IndexError, ValueError):
                    # malformed line - skip or handle as needed
                    pass

            elif current_section == "elements":
                # Example line: 1, 1, 2, 3, 4
                parts = [p.strip() for p in re.split(r",\s*", line)]
                try:
                    # Use element ID (parts[0]) as the key and connectivity nodes as the value
                    element_id = int(parts[0])
                    connectivity = tuple(int(x) for x in parts[1:])
                    data["elements"][element_id] = connectivity  # Store as a dictionary
                except (IndexError, ValueError):
                    pass

            elif current_section == "cloads":
                # Example line: LOAD_SET, 3, -11998.93671200517
                parts = [p.strip() for p in re.split(r",\s*", line)]
                if len(parts) == 3:
                    try:
                        # Only get the force value (last part)
                        force_value = parts[2]
                        data["cloads"]=force_value
                    except ValueError:
                        pass


    return data


def extract_displacements(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    displacements = {}  # node_id: [U1, U2, U3, UR1, UR2, UR3]
    capture = False
    for line in lines:
        if re.search(r'^\s*NODE\s+FOOT-', line):
            capture = True
            continue
        elif capture and (line.strip().startswith("MAXIMUM")):
            break
        elif capture:
            parts = line.strip().split()
            
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    values = parts[1:]
                    displacements[node_id] = values
                except ValueError:
                    continue

    
    return displacements


def main(base_folder="test_data", log_file_path="process_log.txt"):
    results = {}  # Store results per variant folder
    skipped_folders = {}  # Track skipped folders and reasons

    # Open the log file for writing
    with open(log_file_path, "w") as log_file:
        log_file.write("Processing Log\n")
        log_file.write("====================\n")

        for variant_folder in os.listdir(base_folder):
            variant_path = os.path.join(base_folder, variant_folder)
            if not os.path.isdir(variant_path):
                continue  # Skip files if any

            # Initialize placeholders for files
            inp_file = None
            dat_file = None
            log_file_path = None

            # Find .inp, .dat, .log files in variant folder
            for file_name in os.listdir(variant_path):
                if file_name.lower().endswith(".inp"):
                    inp_file = os.path.join(variant_path, file_name)
                elif file_name.lower().endswith(".dat"):
                    dat_file = os.path.join(variant_path, file_name)
                elif file_name.lower().endswith(".log"):
                    log_file_path = os.path.join(variant_path, file_name)

            # Check all required files exist
            if not inp_file or not dat_file or not log_file_path:
                reason = "Missing required files"
                skipped_folders[variant_folder] = reason
                log_file.write(f"Skipped {variant_folder}: {reason}\n")
                continue

            # Check if simulation completed successfully
            simulation_done = is_simulation_completed(dat_file)
            inp_data = None
            displacements = None

            if simulation_done:
                # Parse input file
                inp_data = parse_inp_file_with_re(inp_file)

                # Extract displacements from .dat file
                displacements = extract_displacements(dat_file)

                # Store in results
                results[variant_folder] = {
                    "simulation_completed": simulation_done,
                    "inp_data": inp_data,
                    "displacements": displacements,
                }

            # Log successful processing
            log_file.write(f"Processed {variant_folder}: Simulation Completed = {simulation_done}\n")

        # Log skipped folders
        if skipped_folders:
            log_file.write("\nSkipped Folders:\n")
            for folder, reason in skipped_folders.items():
                log_file.write(f"  {folder}: {reason}\n")

    return results


def save_to_hdf5(results, output_path="meshgraph_dataset.h5"):
    import h5py
    import numpy as np

    with h5py.File(output_path, "w") as h5f:
        for variant, data in results.items():
            nodes = data["inp_data"]["nodes"]
            displacements = data["displacements"]
            cload_val = float(data["inp_data"]["cloads"]) if data["inp_data"]["cloads"] else 0.0

            node_features = []
            node_labels = []

            for node in nodes:
                node_id, x, y, z = node
                disp = displacements.get(node_id)
                if disp and len(disp) >= 6:
                    try:
                        u = list(map(float, disp[:6]))  # U1–UR3
                        applied_cload = 0.0 if node_id in [1, 6] else cload_val
                        node_features.append([x, y, z, applied_cload])
                        node_labels.append(u)
                    except:
                        continue

            if node_features and node_labels:
                node_features = np.array(node_features, dtype=np.float32)  # shape (N, 4)
                node_labels = np.array(node_labels, dtype=np.float32)      # shape (N, 6)

                grp = h5f.create_group(variant)
                grp.create_dataset("node_features", data=node_features)
                grp.create_dataset("displacements", data=node_labels)

    print(f"HDF5 dataset saved at: {output_path}")
    


if __name__ == "__main__":
    
    base_folder = os.path.join("..", "test_data")  
    results = main(base_folder)
    save_to_hdf5(results)
    # Print 1-2 sample variants to verify
    max_print = 2
    for i, (variant, data) in enumerate(results.items()):
        if i >= max_print:
            break

        print(f"\n=== {variant} ===")
        nodes = data["inp_data"]["nodes"]
        displacements = data["displacements"]
        cload_val = float(data["inp_data"]["cloads"]) if data["inp_data"]["cloads"] else 0.0

        print(f"{'NodeID':>6} | {'x':>10} {'y':>10} {'z':>10} | {'cload':>10} || {'U1':>10} {'U2':>10} {'U3':>10} {'UR1':>10} {'UR2':>10} {'UR3':>10}")
        print("-" * 100)

        for node in nodes:
            node_id, x, y, z = node
            disp = displacements.get(node_id)
            if disp:
                try:
                    u1, u2, u3, ur1, ur2, ur3 = map(float, disp[:6])
                    applied_cload = 0.0 if node_id in [1, 6] else cload_val
                    print(f"{node_id:6d} | {x:10.4f} {y:10.4f} {z:10.4f} | {applied_cload:10.2f} || {u1:10.5f} {u2:10.5f} {u3:10.5f} {ur1:10.5f} {ur2:10.5f} {ur3:10.5f}")
                except:
                    continue
