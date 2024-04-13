from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint # Escolher os hiperparametros


test_validation_shape0 = 14000
train_shape0 = 42000

pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

train_labels = openDatasets_l("train_labels.bin", train_shape0)
test_labels = openDatasets_l("test_labels.bin", test_validation_shape0)
validation_labels = openDatasets_l("validation_labels.bin", test_validation_shape0)

# ----------------------------------------------------------------------------------
# TREINAMENTO
# Train sem ruido 
X = pixels_sem_ruido_train/255
X = X.reshape(X.shape[0], -1)
svm_sem_ruido = make_pipeline(LinearSVC(C=3, random_state=42, dual='auto'))
svm_sem_ruido.fit(X, train_labels)

# Train com ruido 
X = pixels_com_ruido_train
X = X.reshape(X.shape[0], -1)
svm_com_ruido = make_pipeline(LinearSVC(C=3, random_state=42, dual='auto'))
svm_com_ruido.fit(X, train_labels)

# # Validation sem ruido 
# X = pixels_sem_ruido_validation/255
# X = X.reshape(X.shape[0], -1)
# svm_sem_ruidoV = make_pipeline(LinearSVC(C=3, random_state=42, dual='auto'))
# svm_sem_ruidoV.fit(X, validation_labels)

# # Validation com ruido 
# X = pixels_com_ruido_validation
# X = X.reshape(X.shape[0], -1)
# svm_com_ruidoV = make_pipeline(LinearSVC(C=3, random_state=42, dual='auto'))
# svm_com_ruidoV.fit(X, validation_labels)

# -----------------------------------------------------------------------------------
# TESTE
# Sem ruido train, Sem ruido teste

print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

n_erros = np.sum(svm_sem_ruido_predicts_1 != test_labels)
taxa_erro_svm_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {svm_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Sem ruido train, Com ruido teste

print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

n_erros = np.sum(svm_sem_ruido_predicts_1 != test_labels)
taxa_erro_svm_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {svm_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Sem ruido teste

print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

n_erros = np.sum(svm_com_ruido_predicts_1 != test_labels)
taxa_erro_svm_com_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {svm_com_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Com ruido teste

print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

n_erros = np.sum(svm_com_ruido_predicts_1 != test_labels)
taxa_erro_svm_com_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {svm_com_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")

# -----------------------------------------------------------------------------------







# # VALIDAÇÃO USANDO COMO TESTE
# # Sem ruido train, Sem ruido validation

# print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Sem Ruido)")
# X = pixels_sem_ruido_validation/255
# X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
# svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

# n_erros = np.sum(svm_sem_ruido_predicts_1 != validation_labels)
# taxa_erro_svm_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

# print(f"Predicts: {svm_sem_ruido_predicts_1}")
# print("Valores Esperados:", validation_labels)
# print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# # -----------------------------------------------------------------------------------
# # Sem ruido train, Com ruido teste

# print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Com Ruido)")
# X = pixels_com_ruido_validation
# X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
# svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

# n_erros = np.sum(svm_sem_ruido_predicts_1 != validation_labels)
# taxa_erro_svm_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

# print(f"Predicts: {svm_sem_ruido_predicts_1}")
# print("Valores Esperados:", validation_labels)
# print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# # -----------------------------------------------------------------------------------
# # Com ruido train, Sem ruido teste

# print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Sem Ruido)")
# X = pixels_sem_ruido_validation/255
# X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
# svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

# n_erros = np.sum(svm_com_ruido_predicts_1 != validation_labels)
# taxa_erro_svm_com_ruido_1 = (n_erros/len(validation_labels)) * 100

# print(f"Predicts: {svm_com_ruido_predicts_1}")
# print("Valores Esperados:", validation_labels)
# print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")

# # -----------------------------------------------------------------------------------
# # Com ruido train, Com ruido teste

# print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Com Ruido)")
# X = pixels_com_ruido_validation
# X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
# svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

# n_erros = np.sum(svm_com_ruido_predicts_1 != validation_labels)
# taxa_erro_svm_com_ruido_1 = (n_erros/len(validation_labels)) * 100

# print(f"Predicts: {svm_com_ruido_predicts_1}")
# print("Valores Esperados:", validation_labels)
# print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")


print("F")
















# # -----------------------------------------------------------------------------------
# # Modelo LN Train
# polynomial_svm_clf = make_pipeline(
# PolynomialFeatures(degree=3),
# StandardScaler(),
# LinearSVC(C=10, max_iter=10_000, random_state=42)
# )
# polynomial_svm_clf.fit(X, train_labels)

# # ------------------------------------------------------------------------------------
# # TESTE LN

# # Sem ruido train, Sem ruido teste
# print("SVM LN (Conjunto de Treino: Sem Ruido) (Conjunto de Teste: Sem Ruido)")
# X = pixels_sem_ruido_test
# X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
# svm_sem_ruido_predicts_1_LN = polynomial_svm_clf.predict(X)

# n_erros = np.sum(svm_sem_ruido_predicts_1_LN != test_labels)
# taxa_erro_svm_sem_ruido_1_LN = (n_erros/len(test_labels)) * 100

# print(f"Predicts: {svm_sem_ruido_predicts_1_LN}")
# print("Valores Esperados:", test_labels)
# print("Taxa de erro:", taxa_erro_svm_sem_ruido_1_LN, "%\n")














# Redimensionamento

# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Suponha que 'data' seja o seu conjunto de dados com dimensões (70000, 28, 28)

# # Redimensionando os dados para (70000, 784) - achatando cada imagem 28x28 em um vetor de 784 elementos
# data = pixels_sem_ruido_train.reshape((42000, 28*28))

# # Aplicando PCA para reduzir a dimensionalidade para 2 componentes
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(data)

# # Plotando os pontos em um gráfico 2D
# plt.scatter(data_pca[:, 0],np.arange(42000), color='blue')
# plt.scatter(data_pca[:, 1], np.arange(42000), color=['red'])
# plt.title('Visualização dos Dados em 2D após PCA')
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.show()

# print("F")

