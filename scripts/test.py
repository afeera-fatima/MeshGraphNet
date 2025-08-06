import h5py

def inspect_hdf5_file(file_path):
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")

    with h5py.File(file_path, "r") as hdf5_file:
        print("HDF5 File Structure:")
        hdf5_file.visititems(print_structure)

# Replace 'your_file.h5' with the path to your HDF5 file
inspect_hdf5_file("meshgraph_dataset.h5")