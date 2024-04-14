import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time

train_and_validation_shape0 = 56000
test_validation_shape0 = 14000
train_shape0 = 42000

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

# -------------------------------------------------------------------------------------
# Encontrar hiperparametros

# Grid
param_grid = {'linearsvc__C': uniform(0.1, 100)}

svm_pipeline = make_pipeline(LinearSVC(random_state=42, dual='auto'))

# Realize a pesquisa aleatória com validação cruzada
random_search = RandomizedSearchCV(svm_pipeline, param_distributions=param_grid, n_iter=2, cv=2, scoring='accuracy', random_state=42)

# Train sem ruido 
X = pixels_sem_ruido_train/255
X = X.reshape(X.shape[0], -1)
start_time = time.time()
random_search.fit(X, train_labels)
end_time = time.time()

# Visualizar os resultados
idealHyperparameter = random_search.best_params_["linearsvc__C"]

print("Melhor valor para max_depth encontrado:", idealHyperparameter)
print("Melhor pontuação de validação cruzada encontrada:", random_search.best_score_*100, "%")

elapsed_time_minutes = (end_time - start_time) / 60
print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))

# ----------------------------------------------------------------------------------
# TREINAMENTO

# Train sem ruido 
X = pixels_sem_ruido_train/255
X = X.reshape(X.shape[0], -1)
svm_sem_ruido = make_pipeline(LinearSVC(random_state=42, dual='auto', C=idealHyperparameter, class_weight='balanced', tol=0.001))
svm_sem_ruido.fit(X, train_labels)

# Train com ruido 
X = pixels_com_ruido_train
X = X.reshape(X.shape[0], -1)
svm_com_ruido = make_pipeline(LinearSVC(random_state=42, dual='auto', C=idealHyperparameter, class_weight='balanced', tol=0.001))
svm_com_ruido.fit(X, train_labels)

# ------------------------------------------------------------------------------------------------------
# VALIDAÇÃO
# Sem ruido train, Sem ruido validation

print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

n_erros = np.sum(svm_sem_ruido_predicts_1 != validation_labels)
taxa_erro_svm_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {svm_sem_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Sem ruido train, Com ruido validation

print("SVM (Conjunto de Treino: Sem Ruido) (Conjunto de Validation: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
svm_sem_ruido_predicts_1 = svm_sem_ruido.predict(X)

n_erros = np.sum(svm_sem_ruido_predicts_1 != validation_labels)
taxa_erro_svm_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {svm_sem_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_svm_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Sem ruido validation

print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

n_erros = np.sum(svm_com_ruido_predicts_1 != validation_labels)
taxa_erro_svm_com_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {svm_com_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Com ruido teste

print("SVM (Conjunto de Treino: Com Ruido) (Conjunto de Validation: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
svm_com_ruido_predicts_1 = svm_com_ruido.predict(X)

n_erros = np.sum(svm_com_ruido_predicts_1 != validation_labels)
taxa_erro_svm_com_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {svm_com_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_svm_com_ruido_1, "%\n")

print("F")