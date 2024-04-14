import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.stats import randint
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
# HIPERPARAMETRO

knn = KNeighborsClassifier()

# Grid
param_dist = {'n_neighbors': randint(1, 50)}  # Intervalo para o parâmetro k
random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=3, cv=3, random_state=42)

# Train sem ruido
X = pixels_sem_ruido_train
X = X.reshape(X.shape[0], -1)
start_time = time.time()
random_search.fit(X, train_labels)
end_time = time.time()

# cross_val_score é igual o randomCV ou manual
#knn_sem_ruido = KNeighborsClassifier(n_neighbors=15)
#knn_sem_ruido.fit(X,train_labels)
#cross_validation = cross_val_score(knn_sem_ruido, X, train_labels, cv=3,scoring="accuracy")

# Melhor valor 
idealHyperparameter = random_search.best_params_["n_neighbors"]
print("Melhor valor de k encontrado:", idealHyperparameter)
# Accuracy
print("Acurácia do modelo com o melhor k:", random_search.best_score_ * 100, "%")

elapsed_time_minutes = (end_time - start_time) / 60
print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))
# ----------------------------------------------------------------------------------- 
# Treinamento para o teste

# Train sem ruido (K = 5)
X = pixels_sem_ruido_train/255
X = X.reshape(X.shape[0], -1)
knn_sem_ruido = KNeighborsClassifier(n_neighbors=idealHyperparameter)
knn_sem_ruido.fit(X,train_labels)

# Train com ruido (K = 5)
X = pixels_com_ruido_train
X = X.reshape(X.shape[0], -1)
knn_com_ruido = KNeighborsClassifier(n_neighbors=idealHyperparameter)
knn_com_ruido.fit(X,train_labels)

#------------------------------------------------------------------------------------
# Rodando os modelos nos conjuntos de validação

# Validation sem ruido train, sem ruido validation
print("KNN com K=5 (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
knn_sem_ruido_predicts_1 = knn_sem_ruido.predict(X)

n_erros = np.sum(knn_sem_ruido_predicts_1 != validation_labels)
taxa_erro_knn_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {knn_sem_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_knn_sem_ruido_1, "%\n")

# ------------------------------------------------------------------------------------
# Validation sem ruido train, com ruido validation
print("KNN com K=5 (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
knn_sem_ruido_predicts_2 = knn_sem_ruido.predict(X)

n_erros = np.sum(knn_sem_ruido_predicts_2 != validation_labels)
taxa_erro_knn_sem_ruido_2 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {knn_sem_ruido_predicts_2}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_knn_sem_ruido_2, "%\n")

# ------------------------------------------------------------------------------------
# Validation com ruido train, sem ruido validation
print("KNN com K=5 (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
knn_com_ruido_predicts_1 = knn_com_ruido.predict(X)

n_erros = np.sum(knn_com_ruido_predicts_1 != validation_labels)
taxa_erro_knn_com_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {knn_com_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_knn_com_ruido_1, "%\n")

# ------------------------------------------------------------------------------------
# Validation com ruido train, sem ruido validation
print("KNN com K=5 (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
knn_com_ruido_predicts_2 = knn_com_ruido.predict(X)

n_erros = np.sum(knn_com_ruido_predicts_2 != validation_labels)
taxa_erro_knn_com_ruido_2 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {knn_com_ruido_predicts_2}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_knn_com_ruido_2, "%\n")