from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(original_image, noisy_image):
    peak_power = 255**2  # Potência de pico
    num_elements = original_image.size  # Número de elementos na imagem

    # Calcular a potência do sinal (imagem original)
    power_x = np.sum(original_image**2) / num_elements

    # Calcular a potência do ruído
    noise = noisy_image - original_image
    power_noise = np.sum(noise**2) / num_elements

    # Calcular o PSNR
    psnr = 10 * np.log10(peak_power / power_noise)
    return psnr

# Carregar o conjunto de dados MNIST
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

# Escolher um exemplo aleatório
example_index = 0
some_digit = X[example_index]

#random_index = np.random.randint(0, len(X))
#original_image = X[random_index]

peak_power = 255**2
# Definir a PSNR desejada
desired_psnr = 10  # PSNR desejada em dB

# Calcular a potência do ruído necessária para atingir a PSNR desejada
noise_power = peak_power / (10**(desired_psnr / 10))

# Gerar ruído com a potência de ruído desejada
noise = np.random.normal(0, np.sqrt(noise_power), some_digit.shape)

# Adicionar ruído à imagem original
noisy_image = some_digit + noise

# Calcular o PSNR entre a imagem original e a imagem ruidosa
psnr = calculate_psnr(some_digit, noisy_image)

print("PSNR obtido:", psnr, "dB")

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
plot_digit(noisy_image, title="Noisy (PSNR=10dB)")
plt.show()

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 =  (y_test == '5')

clf = LinearSVC()
clf.fit(X_train, y_train_5)
a = clf.predict([some_digit])
#a = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print(a)