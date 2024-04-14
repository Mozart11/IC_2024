import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import time

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

# ------------------------------------------------------------------------------------
# HIERPARAMETRO IDEAL
# Definir o modelo de árvore de decisão
tree_idealHyperparameter = DecisionTreeClassifier(random_state=42)

# Definir a grade de hiperparâmetros a serem testados

param_dist = {
    'max_depth': randint(2, 50)  # Profundidade máxima da árvore
}

# Realizar a pesquisa aleatória com validação cruzada
random_search = RandomizedSearchCV(tree_idealHyperparameter, param_distributions=param_dist, n_iter=3, cv=3, scoring='accuracy', random_state=42)
X = pixels_sem_ruido_train/255
X = X.reshape(pixels_sem_ruido_train.shape[0], -1)
start_time = time.time()
random_search.fit(X, train_labels)
end_time = time.time()

# Visualizar os resultados
idealHyperparameter = random_search.best_params_['max_depth']

print("Melhor valor para max_depth encontrado:", idealHyperparameter)
print("Melhor pontuação de validação cruzada encontrada:", random_search.best_score_*100, "%")

elapsed_time_minutes = (end_time - start_time) / 60
print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))

# ----------------------------------------------------------------------------------
# TREINO
X = pixels_sem_ruido_train/255
X = X.reshape(pixels_sem_ruido_train.shape[0], -1)

tree_sem_ruido = DecisionTreeClassifier(max_depth=idealHyperparameter, random_state=42)
tree_sem_ruido.fit(X, train_labels)

X = pixels_com_ruido_train
X = X.reshape(pixels_com_ruido_train.shape[0], -1)

tree_com_ruido = DecisionTreeClassifier(max_depth=idealHyperparameter, random_state=42)
tree_com_ruido.fit(X, train_labels)

# -----------------------------------------------------------------------------------
# TESTE
# Sem ruido train, Sem ruido teste

print("Decision Tree (Conjunto de Treino: Sem Ruido) (Conjunto de Validação: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
tree_sem_ruido_predicts_1 = tree_sem_ruido.predict(X)

n_erros = np.sum(tree_sem_ruido_predicts_1 != validation_labels)
taxa_erro_tree_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {tree_sem_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_tree_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Sem ruido train, Com ruido teste

print("Decision Tree (Conjunto de Treino: Sem Ruido) (Conjunto de Validação: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
tree_sem_ruido_predicts_1 = tree_sem_ruido.predict(X)

n_erros = np.sum(tree_sem_ruido_predicts_1 != validation_labels)
taxa_erro_tree_sem_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {tree_sem_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_tree_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Sem ruido teste

print("Decision Tree (Conjunto de Treino: Com Ruido) (Conjunto de Validação: Sem Ruido)")
X = pixels_sem_ruido_validation/255
X = X.reshape(pixels_sem_ruido_validation.shape[0], -1)
tree_com_ruido_predicts_1 = tree_com_ruido.predict(X)

n_erros = np.sum(tree_com_ruido_predicts_1 != validation_labels)
taxa_erro_tree_com_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {tree_com_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_tree_com_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Com ruido teste

print("Decision Tree (Conjunto de Treino: Com Ruido) (Conjunto de Validação: Com Ruido)")
X = pixels_com_ruido_validation
X = X.reshape(pixels_com_ruido_validation.shape[0], -1)
tree_com_ruido_predicts_1 = tree_com_ruido.predict(X)

n_erros = np.sum(tree_com_ruido_predicts_1 != validation_labels)
taxa_erro_tree_com_ruido_1 = (n_erros/len(validation_labels)) * 100

print(f"Predicts: {tree_com_ruido_predicts_1}")
print("Valores Esperados:", validation_labels)
print("Taxa de erro:", taxa_erro_tree_com_ruido_1, "%\n")

#-----------------------------------------------------------------------------------