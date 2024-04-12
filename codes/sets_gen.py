import matplotlib.pyplot as plt
import numpy as np
 
np.random.seed(333)

def openDatasetsMnist(path):
    with open("../datasets/"+ path, "rb") as f:  # Imagens do conjunto de teste MNIST
        magic_number = int.from_bytes(f.read(4), 'big')  # Metadados (tem que comer os magic_number)
        num_images = int.from_bytes(f.read(4), 'big') 
        num_rows = int.from_bytes(f.read(4), 'big') # 28
        num_cols = int.from_bytes(f.read(4), 'big') # 28 > 1 imagem
        
        hexs_b = (f.read()) # Retorna uma lista de binarios
        pixels = np.frombuffer(hexs_b, dtype=np.uint8) # Transforma a sequencia de bits conforme o dtype, 
        pixels_reshape = pixels.reshape(num_images, num_rows, num_cols) # Unsigned_8 = Pixel Format = 0 a 255 valores

        #plt.imshow(pixels_reshape[0], cmap='gray')  # plt para plotar imagem
        # plt.show()
    return(pixels_reshape)

pixels_sem_ruido_test_MNIST = openDatasetsMnist("t10k-images.idx3-ubyte")
pixels_sem_ruido_train_MNIST = openDatasetsMnist("train-images.idx3-ubyte")

# -------------------------------------------------------------------------------------------
# Conjunto Todo

pixels_sem_ruido_total_MNIST = np.concatenate((pixels_sem_ruido_test_MNIST, pixels_sem_ruido_train_MNIST), axis=0) # Train + Test do MNIST
todos_os_indices = np.arange(pixels_sem_ruido_total_MNIST.shape[0])

# -------------------------------------------------------------------------------------------
# Conjunto de teste sem ruído (20%)

npixels_imgs_teste = 0.2 * len(pixels_sem_ruido_total_MNIST)
indices_aleatorios_teste = np.random.choice(pixels_sem_ruido_total_MNIST.shape[0], int(npixels_imgs_teste), replace=False)

pixels_sem_ruido_teste = pixels_sem_ruido_total_MNIST[indices_aleatorios_teste]

with open("../datasets/test.bin", "wb") as arquivo:
    arquivo.write(pixels_sem_ruido_teste)

# -------------------------------------------------------------------------------------------
# Indices Restantes

indices_restantes = np.setdiff1d(todos_os_indices, indices_aleatorios_teste)

# -------------------------------------------------------------------------------------------
# Conjunto de validação (20%)

npixels_imgs_validation = 0.2 * len(pixels_sem_ruido_total_MNIST)

#indices_aleatorios_validation = np.random.choice(pixels_sem_ruido_restante.shape[0], int(npixels_imgs_validation), replace=False)
indices_aleatorios_validation = np.random.choice(indices_restantes, int(npixels_imgs_validation), replace=False)

pixels_sem_ruido_validation = pixels_sem_ruido_total_MNIST[indices_aleatorios_validation]

with open("../datasets/validation.bin", "wb") as arquivo:
    arquivo.write(pixels_sem_ruido_validation)
# -------------------------------------------------------------------------------------------
# Conjunto Treino (80%)

#indices_treino_validation = np.stack((indices_aleatorios_validation, indices_aleatorios_teste)).reshape(-1)

indices_aleatorios_train = np.setdiff1d(indices_restantes, indices_aleatorios_validation)
pixels_sem_ruido_train = pixels_sem_ruido_total_MNIST[indices_aleatorios_train]

with open("../datasets/train.bin", "wb") as arquivo:
    arquivo.write(pixels_sem_ruido_train)
# -------------------------------------------------------------------------------------------

plt.imshow(pixels_sem_ruido_train[7], cmap='gray')  # plt para plotar imagem
plt.show()
plt.imshow(pixels_sem_ruido_total_MNIST[17], cmap='gray')  # plt para plotar imagem
plt.show()
#plt.imshow(pixels_sem_ruido_restante[4], cmap='gray')  # plt para plotar imagem
#plt.show()