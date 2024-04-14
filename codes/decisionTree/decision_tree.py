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
train_and_validation_shape0 = 56000

pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
pixels_sem_ruido_train_and_validation = openDatasets("train_and_validation.bin", train_and_validation_shape0)
pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_train_and_validation = openDatasets_f("train_and_validation_noise.bin", train_and_validation_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

train_labels = openDatasets_l("train_and_validation_labels.bin", train_shape0)
train_and_validation_labels = openDatasets_l("train_and_validation_labels.bin", train_and_validation_shape0)
test_labels = openDatasets_l("test_labels.bin", test_validation_shape0)
validation_labels = openDatasets_l("validation_labels.bin", test_validation_shape0)


# ----------------------------------------------------------------------------------
# TREINO
X = pixels_sem_ruido_train_and_validation/255
X = X.reshape(pixels_sem_ruido_train_and_validation.shape[0], -1)

tree_sem_ruido = DecisionTreeClassifier(max_depth=16, random_state=42)
tree_sem_ruido.fit(X, train_and_validation_labels)

X = pixels_com_ruido_train_and_validation
X = X.reshape(pixels_com_ruido_train_and_validation.shape[0], -1)

tree_com_ruido = DecisionTreeClassifier(max_depth=16, random_state=42)
tree_com_ruido.fit(X, train_and_validation_labels)

# -----------------------------------------------------------------------------------
# TESTE
# Sem ruido train, Sem ruido teste

print("Tree (Conjunto de Treino Full: Sem Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
tree_sem_ruido_predicts_1 = tree_sem_ruido.predict(X)

n_erros = np.sum(tree_sem_ruido_predicts_1 != test_labels)
taxa_erro_tree_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {tree_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_tree_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Sem ruido train, Com ruido teste

print("Tree (Conjunto de Treino Full: Sem Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
tree_sem_ruido_predicts_1 = tree_sem_ruido.predict(X)

n_erros = np.sum(tree_sem_ruido_predicts_1 != test_labels)
taxa_erro_tree_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {tree_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_tree_sem_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Sem ruido teste

print("Tree (Conjunto de Treino Full: Com Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
tree_com_ruido_predicts_1 = tree_com_ruido.predict(X)

n_erros = np.sum(tree_com_ruido_predicts_1 != test_labels)
taxa_erro_tree_com_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {tree_com_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_tree_com_ruido_1, "%\n")

# -----------------------------------------------------------------------------------
# Com ruido train, Com ruido teste

print("Tree (Conjunto de Treino Full: Com Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
tree_com_ruido_predicts_1 = tree_com_ruido.predict(X)

n_erros = np.sum(tree_com_ruido_predicts_1 != test_labels)
taxa_erro_tree_com_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {tree_com_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_tree_com_ruido_1, "%\n")

#-----------------------------------------------------------------------------------