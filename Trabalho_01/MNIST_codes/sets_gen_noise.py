import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(333)

# Diretorio do script
current_dir = os.path.dirname(os.path.abspath(__file__))

def openDatasets(path, c):  # path e tipo de conjunto (train,val ou test)
    with open(current_dir + "/../MNIST_datasets/" + path, "rb") as f:
        bins = f.read()
        pixels = np.frombuffer(bins, dtype=np.uint8)
        pixels_reshape = pixels.reshape(c, 28, 28)
        return(pixels_reshape)

def openDatasets_f(path, c):  # path e tipo de conjunto (train,val ou test)
    with open(current_dir + "/../MNIST_datasets/" + path, "rb") as f:
        bins = f.read()
        pixels = np.frombuffer(bins, dtype=np.float64)
        pixels_reshape = pixels.reshape(c, 28, 28)
        return(pixels_reshape)

def openDatasets_l(path, c):  # path e tipo de conjunto (train,val ou test)
    with open(current_dir + "/../MNIST_datasets/" + path, "rb") as f:
        bins = f.read()
        pixels = np.frombuffer(bins, dtype=np.uint8)
        return(pixels)

def gen_gaussia_noise(var,x,y):
    sigma = np.sqrt(var)
    n = np.random.normal(loc=0,
                        scale=sigma,
                        size=(x,y))
    return n

def add_noise(noise,image):
    # corrupt image = original image + gaussian noise
    corrupt = image + noise
    return corrupt

def find_variance(img,target_psnr):
    # sigma^2 == MSE
    sigma = np.amax(img)/ np.power(10,target_psnr/10)
    return sigma

def set_noise_gen(pixels, pixelsNoise):
    for i in range(pixels.shape[0]):
        img = pixels[i]
        img = img / 255  # Porque ? > Normaliza os valores entre 0 e 1
        #img = img)            #.astype(np.float64)
        x,y = img.shape
        #psnr = float(input("PSNR: "))
        psnr = 10

        sigma = find_variance(img,psnr)
        noise = gen_gaussia_noise(sigma,x,y)
        corrupt_img = add_noise(noise,img)
        pixelsNoise[i] = corrupt_img
    return pixelsNoise

if __name__ == "__main__":
    test_validation_shape0 = 14000
    train_shape0 = 42000
    train_and_validation_shape0 = 56000

    pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
    pixels_sem_ruido_train_and_validation = openDatasets("train_and_validation.bin", train_and_validation_shape0)
    pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
    pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

    pixels_com_ruido_train_empty = np.empty_like(pixels_sem_ruido_train, dtype=np.float64)
    pixels_com_ruido_train_and_validation_empty = np.empty_like(pixels_sem_ruido_train_and_validation, dtype=np.float64)
    pixels_com_ruido_test_empty = np.empty_like(pixels_sem_ruido_test, dtype=np.float64)
    pixels_com_ruido_validation_empty = np.empty_like(pixels_sem_ruido_validation, dtype=np.float64)

    pixels_com_ruido_train = set_noise_gen(pixels_sem_ruido_train, pixels_com_ruido_train_empty)
    pixels_com_ruido_train_and_validation = set_noise_gen(pixels_sem_ruido_train_and_validation, pixels_com_ruido_train_and_validation_empty)
    pixels_com_ruido_test = set_noise_gen(pixels_sem_ruido_test, pixels_com_ruido_test_empty)
    pixels_com_ruido_validation = set_noise_gen(pixels_sem_ruido_validation, pixels_com_ruido_validation_empty)

    with open(current_dir + "/../MNIST_datasets/train_noise.bin", "wb") as arquivo:
        arquivo.write(pixels_com_ruido_train)            #.astype(np.uint8))
    with open(current_dir +"/../MNIST_datasets/train_and_validation_noise.bin", "wb") as arquivo:
        arquivo.write(pixels_com_ruido_train_and_validation)            #.astype(np.uint8))
    with open(current_dir + "/../MNIST_datasets/test_noise.bin", "wb") as arquivo:
        arquivo.write(pixels_com_ruido_test)            #.astype(np.uint8))
    with open(current_dir + "/../MNIST_datasets/validation_noise.bin", "wb") as arquivo:
        arquivo.write(pixels_com_ruido_validation)            #.astype(np.uint8))
