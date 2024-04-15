import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.datasets import fetch_olivetti_faces
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from scipy.stats import uniform

np.random.seed(333)

# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784')

# Target variable is the median house value for California districts,\nexpressed in hundreds of thousands of dollars ($100,000)
#print(california_housing)

# Load the dataset
faces = fetch_olivetti_faces()

# Get the data and target labels
X = faces.data
X = X.reshape(400,64,64)
y = faces.target

# Shuffle
indices_embaralhados = np.random.permutation(len(X))
X_embaralhado = X[indices_embaralhados]
y_embaralhado = y[indices_embaralhados]
#print(faces)

def retorna_porcentagem_train_test(porcentagem):
    train = X_embaralhado[:int((porcentagem/100) * len(X))]
    train_labels = y_embaralhado[:int((porcentagem/100) * len(y))]
    test = X_embaralhado[int((porcentagem/100) * len(X)):]
    test_labels = y_embaralhado[int((porcentagem/100) * len(y)):]
    return train, train_labels, test, test_labels

train, train_labels, test, test_labels = retorna_porcentagem_train_test(70) # 70% TRAIN, 30% TEST

# # ------------------------------------------------------------------------------------
# # Hiperparametro Ideal #### Problema de numero de iterações

# # Parameters
# param_grid = {
#     'linearsvc__C': uniform(0.1, 100),
#     'linearsvc__class_weight': [None, 'balanced']  # Adicionando o parâmetro class_weight
# }
# # -----------------------------------------------------------------------------------------------

# # train (hiperparametro)
# svm_pipeline = make_pipeline(LinearSVC(random_state=42, dual='auto'))
# random_search = RandomizedSearchCV(svm_pipeline, param_distributions=param_grid, n_iter=30, cv=3, scoring='accuracy', random_state=42)

X = train
X = X.reshape(X.shape[0], -1)

# start_time = time.time()
# random_search.fit(X, train_labels)
# end_time = time.time()

# # Visualizar os resultados
# idealHyperparameter_1 = random_search.best_params_["linearsvc__C"]
# idealHyperparameter_2 = random_search.best_params_["linearsvc__class_weight"]
# print("Melhor valor para linearsvc__C encontrado:", idealHyperparameter_1)
# print("Melhor valor para linearsvc__class_weight encontrado:", idealHyperparameter_2)
# print("Melhor pontuação de validação cruzada encontrada:", random_search.best_score_*100, "%")

# elapsed_time_minutes = (end_time - start_time) / 60
# print("Tempo decorrido: {:.2f} minutos".format(elapsed_time_minutes))

# ---------------------------------------------------------------------------------------
# Treino
svm = make_pipeline(LinearSVC(random_state=42, dual='auto', C=1)) #class_weight="balanced"))
svm.fit(X, train_labels)

# Teste
Xt = test
Xt = Xt.reshape(120,-1)
svm_predicts = svm.predict(Xt)

n_erros = np.sum(svm_predicts != test_labels)
taxa_erro_svm = (n_erros/len(test_labels)) * 100

print(f"Predicts: {svm_predicts}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_svm, "%\n")

print("F")
