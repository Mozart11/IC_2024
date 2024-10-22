#Redimensionamento
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from codes.sets_gen_noise import openDatasets, openDatasets_l

train_shape0 = 42000

# Suponha que 'data' seja o seu conjunto de dados com dimensões (70000, 28, 28)
pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
# Redimensionando os dados para (70000, 784) - achatando cada imagem 28x28 em um vetor de 784 elementos
data = pixels_sem_ruido_train.reshape((42000, 28*28))

# Aplicando PCA para reduzir a dimensionalidade para 2 componentes
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Plotando os pontos em um gráfico 2D
plt.scatter(data_pca[:, 0],np.arange(42000), color='blue')
plt.scatter(data_pca[:, 1], np.arange(42000), color=['red'])
plt.title('Visualização dos Dados em 2D após PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

print("F")