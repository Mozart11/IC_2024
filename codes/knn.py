import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sets_gen_noise import openDatasets, openDatasets_l, openDatasets_f

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
# ------------------------------------------------------------------------------------ 
# Treinamento para o teste

# Train sem ruido (K = 5)
X = pixels_sem_ruido_train/255
X = X.reshape(X.shape[0], -1)
knn_sem_ruido = KNeighborsClassifier(n_neighbors=5)
knn_sem_ruido.fit(X,train_labels)

# Train com ruido (K = 5)
X = pixels_com_ruido_train
X = X.reshape(X.shape[0], -1)
knn_com_ruido = KNeighborsClassifier(n_neighbors=5)
knn_com_ruido.fit(X,train_labels)

# ------------------------------------------------------------------------------------
# Treinamento para a validação

# Train sem ruido (K = 5)
# X = pixels_sem_ruido_validation/255
# X = X.reshape(X.shape[0], -1)
# knn_sem_ruidoV = KNeighborsClassifier(n_neighbors=5)
# knn_sem_ruidoV.fit(X,validation_labels)

# # Train com ruido (K = 5)
# X = pixels_com_ruido_validation
# X = X.reshape(X.shape[0], -1)
# knn_com_ruidoV = KNeighborsClassifier(n_neighbors=5)
# knn_com_ruidoV.fit(X,validation_labels)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Rodando os modelos nos conjuntos de teste

#  Sem ruido train, sem ruido test
print("KNN com K=5 (Conjunto de Treino: Sem Ruido) (Conjunto de Teste: Sem Ruido)")
X = pixels_sem_ruido_test/255
X = X.reshape(pixels_sem_ruido_test.shape[0], -1)
knn_sem_ruido_predicts_1 = knn_sem_ruido.predict(X)

n_erros = np.sum(knn_sem_ruido_predicts_1 != test_labels)
taxa_erro_knn_sem_ruido_1 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_sem_ruido_predicts_1}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_sem_ruido_1, "%\n")

# ------------------------------------------------------------------------------------
# Test sem ruido train, com ruido test
print("KNN com K=5 (Conjunto de Treino: Sem Ruido) (Conjunto de Teste: Com Ruido)")
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
print("KNN com K=5 (Conjunto de Treino: Com Ruido) (Conjunto de Teste: Sem Ruido)")
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
print("KNN com K=5 (Conjunto de Treino: Com Ruido) (Conjunto de Teste: Com Ruido)")
X = pixels_com_ruido_test
X = X.reshape(pixels_com_ruido_test.shape[0], -1)
knn_com_ruido_predicts_2 = knn_com_ruido.predict(X)

n_erros = np.sum(knn_com_ruido_predicts_2 != test_labels)
taxa_erro_knn_com_ruido_2 = (n_erros/len(test_labels)) * 100

print(f"Predicts: {knn_com_ruido_predicts_2}")
print("Valores Esperados:", test_labels)
print("Taxa de erro:", taxa_erro_knn_com_ruido_2, "%\n")

# -----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
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