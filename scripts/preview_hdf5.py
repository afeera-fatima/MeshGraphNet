import os
import h5py
import numpy as np

def preview_hdf5_structure(h5_path="meshgraph_dataset_dgl.h5"):
    """
    Preview the structure of the HDF5 file and show what datasets are available.
    """
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return
    
    print(f"📁 HDF5 File: {h5_path}")
    print("=" * 80)
    
    with h5py.File(h5_path, "r") as f:
        def print_structure(name, obj):
            indent = "  " * (name.count('/') - 1) if name.count('/') > 0 else ""
            if isinstance(obj, h5py.Group):
                print(f"{indent}📂 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}, Data Type: {obj.dtype}")
        
        print("HDF5 File Structure:")
        f.visititems(print_structure)
        
        # Count variants
        variant_count = len([key for key in f.keys() if f[key]])
        print(f"\n📊 Total Variants: {variant_count}")

def preview_variant_data(h5_path="meshgraph_dataset_dgl.h5", variant_name=None, max_nodes=10, max_edges=10):
    """
    Preview the actual data from a specific variant.
    """
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return
    
    with h5py.File(h5_path, "r") as f:
        # Get the first variant if none specified
        if variant_name is None:
            variant_names = list(f.keys())
            if not variant_names:
                print("❌ No variants found in the file")
                return
            variant_name = variant_names[0]
        
        if variant_name not in f:
            print(f"❌ Variant '{variant_name}' not found in the file")
            available = list(f.keys())[:10]
            print(f"Available variants (first 10): {available}")
            return
        
        print(f"\n🔍 Previewing Variant: {variant_name}")
        print("=" * 80)
        
        # Access the case_000 group within the variant
        case_group = f[f"{variant_name}/case_000"]
        
        # Load datasets
        pos = case_group["pos"][:]
        spc = case_group["spc"][:]
        load = case_group["load"][:]
        y = case_group["y"][:]
        connectivity = case_group["connectivity"][:]
        
        print(f"📊 Data Summary:")
        print(f"   Nodes: {len(pos)}")
        print(f"   Edges: {len(connectivity)}")
        print(f"   Node Features Dimensions:")
        print(f"     - pos (positions): {pos.shape}")
        print(f"     - spc (constraints): {spc.shape}")
        print(f"     - load (forces): {load.shape}")
        print(f"     - y (displacements): {y.shape}")
        print(f"   Edge Features Dimensions:")
        print(f"     - connectivity: {connectivity.shape}")
        
        print(f"\n📍 Node Data (showing first {min(max_nodes, len(pos))} nodes):")
        print(f"{'Node':>4} | {'Position (x,y,z)':>25} | {'Constraints':>15} | {'Load':>15} | {'Displacement':>15}")
        print("-" * 85)
        
        for i in range(min(max_nodes, len(pos))):
            pos_str = f"({pos[i][0]:7.3f}, {pos[i][1]:7.3f})"
            if len(pos[i]) > 2:
                pos_str = f"({pos[i][0]:7.3f}, {pos[i][1]:7.3f}, {pos[i][2]:7.3f})"
            
            spc_str = f"({spc[i][0]:4.1f}, {spc[i][1]:4.1f})"
            load_str = f"({load[i][0]:7.2f}, {load[i][1]:7.2f})"
            y_str = f"({y[i][0]:7.3e}, {y[i][1]:7.3e})"
            
            print(f"{i:4d} | {pos_str:>25} | {spc_str:>15} | {load_str:>15} | {y_str:>15}")
        
        if len(pos) > max_nodes:
            print(f"     ... and {len(pos) - max_nodes} more nodes")
        
        print(f"\n🔗 Connectivity Data (showing first {min(max_edges, len(connectivity))} edges):")
        print(f"{'Edge':>4} | {'From Node':>9} | {'To Node':>7}")
        print("-" * 25)
        
        for i in range(min(max_edges, len(connectivity))):
            print(f"{i:4d} | {connectivity[i][0]:9d} | {connectivity[i][1]:7d}")
        
        if len(connectivity) > max_edges:
            print(f"     ... and {len(connectivity) - max_edges} more edges")

def preview_statistics(h5_path="meshgraph_dataset_dgl.h5"):
    """
    Show statistical summary of all variants in the dataset.
    """
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return
    
    print(f"\n📈 Dataset Statistics")
    print("=" * 80)
    
    node_counts = []
    edge_counts = []
    displacement_ranges = []
    load_ranges = []
    
    with h5py.File(h5_path, "r") as f:
        variant_names = list(f.keys())
        
        for variant_name in variant_names:
            try:
                case_group = f[f"{variant_name}/case_000"]
                
                pos = case_group["pos"][:]
                y = case_group["y"][:]
                load = case_group["load"][:]
                connectivity = case_group["connectivity"][:]
                
                node_counts.append(len(pos))
                edge_counts.append(len(connectivity))
                
                # Calculate displacement ranges
                y_flat = y.flatten()
                displacement_ranges.append((np.min(y_flat), np.max(y_flat)))
                
                # Calculate load ranges
                load_flat = load.flatten()
                load_ranges.append((np.min(load_flat), np.max(load_flat)))
                
            except Exception as e:
                print(f"⚠️ Error processing variant {variant_name}: {e}")
                continue
    
    if node_counts:
        print(f"📊 Node Statistics:")
        print(f"   Count: {len(node_counts)} variants")
        print(f"   Nodes per variant: min={min(node_counts)}, max={max(node_counts)}, avg={np.mean(node_counts):.1f}")
        
        print(f"\n🔗 Edge Statistics:")
        print(f"   Edges per variant: min={min(edge_counts)}, max={max(edge_counts)}, avg={np.mean(edge_counts):.1f}")
        
        print(f"\n📏 Displacement Statistics:")
        all_disp_min = min([d[0] for d in displacement_ranges])
        all_disp_max = max([d[1] for d in displacement_ranges])
        print(f"   Global range: {all_disp_min:.6e} to {all_disp_max:.6e}")
        
        print(f"\n⚡ Load Statistics:")
        all_load_min = min([l[0] for l in load_ranges])
        all_load_max = max([l[1] for l in load_ranges])
        print(f"   Global range: {all_load_min:.6e} to {all_load_max:.6e}")

def main():
    """
    Main function to run all preview functions.
    """
    h5_path = "meshgraph_dataset_dgl.h5"
    
    print("🔍 HDF5 Dataset Preview Tool")
    print("=" * 80)
    
    # 1. Show file structure
    preview_hdf5_structure(h5_path)
    
    # 2. Preview first variant data
    preview_variant_data(h5_path, max_nodes=10, max_edges=10)
    
    # 3. Show statistics
    preview_statistics(h5_path)
    
    print("\n✅ Preview completed!")

if __name__ == "__main__":
    main()



####Test file
# import h5py

# def inspect_hdf5_file(file_path):
#     def print_structure(name, obj):
#         if isinstance(obj, h5py.Group):
#             print(f"Group: {name}")
#         elif isinstance(obj, h5py.Dataset):
#             print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")

#     with h5py.File(file_path, "r") as hdf5_file:
#         print("HDF5 File Structure:")
#         hdf5_file.visititems(print_structure)

# # Replace 'your_file.h5' with the path to your HDF5 file
# inspect_hdf5_file("meshgraph_dataset.h5")