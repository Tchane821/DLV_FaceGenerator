import os

from keras import models
from numpy import load
from gans_tools import plot_faces, load_real_samples, define_gan, define_generator, define_discriminator, gan_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load dataset
print("Log: Load Data: In progress...")
path_to_data = "./data"
data = load(f"{path_to_data}/img_align_celeba.npz")
print("Log: Load data: Done!")

# Print shape of one data array
faces = data['arr_0']
print(f"\tInfo: Images loaded = {faces.shape[0]}, size = ({faces.shape[1]},{faces.shape[2]}), depth = {faces.shape[3]}")
# plot_faces(faces, 5)

# Size of the latent space
latent_dim = 100
# Create the discriminator
d_model = define_discriminator()
# Create the generator
g_model = define_generator(latent_dim)
# Create the gan
gan_model = define_gan(g_model, d_model)
# Load image data
dataset = load_real_samples(faces)
# Train or load model
model_name = "generator_model.h5"
if os.path.exists(f"{path_to_data}/{model_name}"):
    print("Log: Model found")
    gan_model = models.load_model(f"{path_to_data}/{model_name}")
    print("Log: Model load")
else:
    print("Log: Model not found training launch...")
    gan_train(g_model, d_model, gan_model, dataset, latent_dim)
    # Save modele
    gan_model.save(f"{path_to_data}/generator_model.h5")
    print("Log: Model trained and saved!")
