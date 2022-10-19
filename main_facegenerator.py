import os

from keras import models
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from numpy import load
from gans_tools import load_real_samples, define_gan, define_generator, define_discriminator, gan_train
from gans_tools import generate_real_samples, generate_fake_samples, scale_images, calculate_fid

from gans_tools import plot_faces

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

# Modele d'évaluation
print("Log: Loading InceptionV3 model: In Progresse...")
eval_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
print("Log: Loading InceptionV3 model: Done!")

# load real and fake images
n_samples = 1000
print("Log: Generating fake and real samples: In progress...")
X_real, _ = generate_real_samples(dataset, n_samples)
X_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
print("Log: Generating fake and real samples: Done!")
# scale from [-1,1] to [0,255]
X_real = X_real * 127.5 + 127.5
X_fake = X_fake * 127.5 + 127.5
plot_faces(X_fake, 5)
# resize images
X_real = scale_images(X_real, (299, 299, 3))
X_fake = scale_images(X_fake, (299, 299, 3))
plot_faces(X_fake, 5)
# pre-process images
X_real = preprocess_input(X_real)
X_fake = preprocess_input(X_fake)
# fid between real and generated images
print("Log: Calculating FID: In progress...")
fid = calculate_fid(eval_model, X_real, X_fake)
print("Log: Calculating FID: Done!")
print(f"\tInfo: FID = {fid}")
