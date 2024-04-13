from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sets_gen_noise import openDatasets

train = openDatasets("train.bin", 42000)

# Função para adicionar ruído com PSNR específico
def add_noise_with_psnr(image_data, psnr_dB):
    # Calcular a variância do ruído com base no PSNR
    sigma = np.std(image_data)
    noise_std = sigma / (10**(psnr_dB / 20))
    # Adicionar ruído gaussiano aos dados da imagem
    noisy_image = image_data + np.random.normal(scale=noise_std, size=image_data.shape)
    return noisy_image

# Carregar o conjunto de dados MNIST
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

# Escolher um exemplo aleatório
example_index = 5
some_digit = X[example_index]
some_digit = train[0]

# Adicionar ruído ao exemplo
noisy_digit = add_noise_with_psnr(some_digit, psnr_dB=10)

# Função para plotar dígitos
def plot_digit(image_data, title):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.title(title)
    plt.axis("off")

# Plotar o dígito original e o dígito com ruído
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_digit(some_digit, title="Original")
plt.subplot(1, 2, 2)
plot_digit(noisy_digit, title="Noisy (PSNR=10dB)")
plt.show()

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]