from numpy import load

# Load dataset
print("Log: Load Data: In progress...")
path_to_data = "./data"
data = load(f"{path_to_data}/img_align_celeba.npz")
print("Log: Load data: Done!")

# Print shape of one data array
faces = data['arr_0']
print(f"\tInfo: Images loaded = {faces.shape[0]}, size = ({faces.shape[1]},{faces.shape[2]}), depth = {faces.shape[3]}")
