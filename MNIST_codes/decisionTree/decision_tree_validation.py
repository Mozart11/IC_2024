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
from scipy.stats import randint, uniform
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

def retorna_porcentagem_sr(porcentagem):
    train_curve_1 = pixels_sem_ruido_train[:int((porcentagem/100) * len(pixels_sem_ruido_train))]
    train_curve_labels_1 = train_labels[:int((porcentagem/100) * len(train_labels))]
    return train_curve_1, train_curve_labels_1

def retorna_porcentagem_cr(porcentagem):
    train_curve_1 = pixels_com_ruido_train[:int((porcentagem/100) * len(pixels_com_ruido_train))]
    train_curve_labels_1 = train_labels[:int((porcentagem/100) * len(train_labels))]
    return train_curve_1, train_curve_labels_1

# ------------------------------------------------------------------------------------
# HIERPARAMETRO IDEAL

# Definir a grade de hiperparâmetros a serem testados e a %
porcentagem = 30 # %
param_dist = {
    'max_depth': randint(1, 20),  # Profundidade máxima da árvore
    'min_samples_leaf': randint(1, 20),  # Mínimo de amostras necessárias para ser um nó folha
}

# ----------------------------------------------------------------------------------------
# TRAIN SEM RUIDO

tree_idealHyperparameter = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(tree_idealHyperparameter, param_distributions=param_dist, n_iter=25, cv=5, scoring='accuracy', random_state=42)

X = retorna_porcentagem_sr(porcentagem)[0]/255
X = X.reshape(X.shape[0], -1)

start_time = time.time()
random_search.fit(X, retorna_porcentagem_sr(porcentagem)[1])
end_time = time.time()

# Visualizar os resultados
idealHyperparameter_1 = random_search.best_params_['max_depth']
idealHyperparameter_2 = random_search.best_params_['min_samples_leaf']

print("Melhor valor para max_depth encontrado:", idealHyperparameter_1)
print("Melhor valor para min_samples_leaf encontrado:", idealHyperparameter_2)
print("Melhor pontuação de validação cruzada encontrada:", random_search.best_score_*100, "%")

elapsed_time_minutes = (end_time - start_time) / 60
print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))

# ----------------------------------------------------------------------------------------
# TRAIN COM RUIDO ################################################ COMPUTACIONALMENTE NÂO COMPENSA

tree_idealHyperparameter = DecisionTreeClassifier(random_state=42)
random_search2 = RandomizedSearchCV(tree_idealHyperparameter, param_distributions=param_dist, n_iter=25, cv=5, scoring='accuracy', random_state=42)

X = retorna_porcentagem_cr(porcentagem)[0]
X = X.reshape(X.shape[0], -1)

start_time = time.time()
random_search2.fit(X, retorna_porcentagem_cr(porcentagem)[1])
end_time = time.time()

# Visualizar os resultados
idealHyperparameter_1 = random_search2.best_params_['max_depth']
idealHyperparameter_2 = random_search2.best_params_['min_samples_leaf']

print("Melhor valor para max_depth encontrado:", idealHyperparameter_1)
print("Melhor valor para min_samples_leaf encontrado:", idealHyperparameter_2)
print("Melhor pontuação de validação cruzada encontrada:", random_search2.best_score_*100, "%")

elapsed_time_minutes = (end_time - start_time) / 60
print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))

# ----------------------------------------------------------------------------------
# TREINO
X = pixels_sem_ruido_train/255
X = X.reshape(pixels_sem_ruido_train.shape[0], -1)

tree_sem_ruido = DecisionTreeClassifier(max_depth=idealHyperparameter_1, min_samples_leaf=idealHyperparameter_2, random_state=42)
tree_sem_ruido.fit(X, train_labels)

X = pixels_com_ruido_train
X = X.reshape(pixels_com_ruido_train.shape[0], -1)

tree_com_ruido = DecisionTreeClassifier(max_depth=idealHyperparameter_1, min_samples_leaf=idealHyperparameter_2, random_state=42)
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