import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f

test_validation_shape0 = 14000
train_shape0 = 42000
train_and_validation_shape0 = 56000

pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
pixels_sem_ruido_train_and_validation = openDatasets("train_and_validation.bin", train_and_validation_shape0)
pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_train_and_validation = openDatasets_f("train_and_validation_noise.bin", train_and_validation_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

train_labels = openDatasets_l("train_labels.bin", train_shape0)
train_and_validation_labels = openDatasets_l("train_and_validation_labels.bin", train_and_validation_shape0)
test_labels = openDatasets_l("test_labels.bin", test_validation_shape0)
validation_labels = openDatasets_l("validation_labels.bin", test_validation_shape0)

# ------------------------------------------------------------------------------------
# Treinamento para o teste

# Train sem ruido (K = 15)
X = pixels_sem_ruido_train_and_validation/255
X = X.reshape(X.shape[0], -1)
knn_sem_ruido = KNeighborsClassifier(n_neighbors=1)
knn_sem_ruido.fit(X,train_and_validation_labels)

# Train com ruido (K = 15)
X = pixels_com_ruido_train_and_validation
X = X.reshape(X.shape[0], -1)
knn_com_ruido = KNeighborsClassifier(n_neighbors=6)
knn_com_ruido.fit(X,train_and_validation_labels)

# -----------------------------------------------------------------------------------
# Rodando os modelos nos conjuntos de teste

#  Sem ruido train, sem ruido test
print("KNN com K=1 (Conjunto de Treino Full: Sem Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
knn_sem_ruido_predicts_1 = knn_sem_ruido.predict(X)

n_erros = np.sum(knn_sem_ruido_predicts_1 != test_labels)
taxa_erro_knn_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_sem_ruido_1, "%\n")

# ------------------------------------------------------------------------------------
# Visualizações dos dados 
sample_weight = (knn_sem_ruido_predicts_1 != test_labels)
#plt.figure(figsize=[10,10])
#plt.subplot(2,2,1)
cm = ConfusionMatrixDisplay.from_predictions(test_labels, knn_sem_ruido_predicts_1, normalize="true", values_format=".0%", sample_weight=sample_weight)
#plt.show()
#plt.subplot(2,2,2)
cm = ConfusionMatrixDisplay.from_predictions(test_labels, knn_sem_ruido_predicts_1, normalize="true", values_format=".0%")
#plt.show()
#plt.subplot(2,2,3)
cm = ConfusionMatrixDisplay.from_predictions(test_labels, knn_sem_ruido_predicts_1)
plt.show()

# ------------------------------------------------------------------------------------
# Test sem ruido train, com ruido test
print("KNN com K=1 (Conjunto de Treino Full: Sem Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
knn_sem_ruido_predicts_2 = knn_sem_ruido.predict(X)

n_erros = np.sum(knn_sem_ruido_predicts_2 != test_labels)
taxa_erro_knn_sem_ruido_2 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_sem_ruido_predicts_2}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_sem_ruido_2, "%\n")

# ------------------------------------------------------------------------------------
# Test com ruido train, sem ruido test
print("KNN com K=6 (Conjunto de Treino Full: Com Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
knn_com_ruido_predicts_1 = knn_com_ruido.predict(X)

n_erros = np.sum(knn_com_ruido_predicts_1 != test_labels)
taxa_erro_knn_com_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_com_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_com_ruido_1, "%\n")

# ------------------------------------------------------------------------------------
# Test com ruido train, sem ruido test
print("KNN com K=6 (Conjunto de Treino Full: Com Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
knn_com_ruido_predicts_2 = knn_com_ruido.predict(X)

n_erros = np.sum(knn_com_ruido_predicts_2 != test_labels)
taxa_erro_knn_com_ruido_2 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_com_ruido_predicts_2}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_com_ruido_2, "%\n")

