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

np.random.seed(333)

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

def retorna_porcentagem(porcentagem):
    train_curve_1 = train[:int((porcentagem/100) * len(train))]
    train_curve_labels_1 = train_labels[:int((porcentagem/100) * len(train_labels))]
    return train_curve_1, train_curve_labels_1

# ------------------------------------------------------------------------------------
# Treinamento para o teste
porcentagem = 5
Xt = test.reshape(test.shape[0], -1)
accuracyValuesArray = []
sampleValues = []
stdValues = []
for i in range(20):
    # Train sem ruido 
    X = retorna_porcentagem(porcentagem)[0]
    X = X.reshape(X.shape[0], -1)
    y = retorna_porcentagem(porcentagem)[1]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X,y)
    
    # Test sem ruido
    knn_predicts = knn.predict(Xt)
    n_erros = np.sum(knn_predicts != test_labels)
    taxa_erro_knn = 100 - ((n_erros/len(test_labels)) * 100)
    
    print(f"KNN com ", porcentagem ,"% do conjunto de treino, K=1")
    print(f"Predicts: {knn_predicts}")
    print("Valores Esperados:", test_labels)
    print("Taxa de acerto:", taxa_erro_knn, "%")
    sampleValues.append((porcentagem/100) * 400)
    accuracyValuesArray.append(taxa_erro_knn) 
    #std = np.std(accuracyValuesArray)
    #stdValues.append(std)
    #print("Standard Deviation: ", std, "\n" )
    #print("Standard Deviation: ", np.std(accuracyValuesArray), "\n" )
    porcentagem += 5

np.savez("./sample_complexity_curve.npz", train=accuracyValuesArray, samples=sampleValues, std=stdValues)
# -----------------------------------------------------------------------------------




