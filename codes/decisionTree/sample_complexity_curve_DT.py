import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f

test_validation_shape0 = 14000
train_shape0 = 42000
train_and_validation_shape0 = 56000

#pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
pixels_sem_ruido_train_and_validation = openDatasets("train_and_validation.bin", train_and_validation_shape0)
pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
#pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

#pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_train_and_validation = openDatasets_f("train_and_validation_noise.bin", train_and_validation_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
#pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

#train_labels = openDatasets_l("train_labels.bin", train_shape0)
train_and_validation_labels = openDatasets_l("train_and_validation_labels.bin", train_and_validation_shape0)
test_labels = openDatasets_l("test_labels.bin", test_validation_shape0)
#validation_labels = openDatasets_l("validation_labels.bin", test_validation_shape0)

# Calcula as % dos conjuntos de treino
def retorna_porcentagem(porcentagem):
    train_curve_1 = pixels_sem_ruido_train_and_validation[:int((porcentagem/100) * len(pixels_com_ruido_train_and_validation))]
    train_curve_labels_1 = train_and_validation_labels[:int((porcentagem/100) * len(pixels_com_ruido_train_and_validation))]
    return train_curve_1, train_curve_labels_1

# ------------------------------------------------------------------------------------
# Treinamento para o teste
porcentagem = 5
Xt = pixels_sem_ruido_test.reshape(pixels_sem_ruido_test.shape[0], -1)/255
accuracyValuesArray = []
sampleValues = []
for i in range(20):
    # Train sem ruido 
    X = retorna_porcentagem(porcentagem)[0]/255
    X = X.reshape(X.shape[0], -1)
    y = retorna_porcentagem(porcentagem)[1]
    tree_sem_ruido = DecisionTreeClassifier(max_depth=16, random_state=42)
    tree_sem_ruido.fit(X,y)
    
    # Test sem ruido
    tree_sem_ruido_predicts = tree_sem_ruido.predict(Xt)
    n_erros = np.sum(tree_sem_ruido_predicts != test_labels)
    taxa_erro_tree_sem_ruido = 100 - ((n_erros/len(test_labels)) * 100)

    print(f"KNN com ", porcentagem ,"% do conjunto de treino, K=15")
    print(f"Predicts: {tree_sem_ruido_predicts}")
    print("Valores Esperados:", test_labels)
    print("Taxa de acerto:", taxa_erro_tree_sem_ruido, "%\n")
    sampleValues.append((porcentagem/100) * 56000)
    accuracyValuesArray.append(taxa_erro_tree_sem_ruido) 
    porcentagem += 5

np.savez("./sample_complexity_curve.npz", train=accuracyValuesArray, samples=sampleValues)
# -----------------------------------------------------------------------------------




