import sys
import os

# Adicionar o diretório atual ao PYTHONPATH
cwd = os.getcwd() + "/.."
sys.path.append(cwd)

import numpy as np
from sets_gen_noise import openDatasets, openDatasets_f, openDatasets_l

test_validation_shape0 = 14000
train_shape0 = 42000

pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)
pixels_sem_ruido_test = openDatasets("test.bin", test_validation_shape0)
pixels_sem_ruido_validation = openDatasets("validation.bin", test_validation_shape0)

pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

# Verificação de repetição dos indices
# Verificar se há números repetidos entre quaisquer dois dos três arrays
repetidos_12 = np.intersect1d(pixels_sem_ruido_train, pixels_sem_ruido_train.astype(np.float64))
#repetidos_13 = np.intersect1d(indices_aleatorios_teste, indices_aleatorios_train)
#repetidos_23 = np.intersect1d(indices_aleatorios_train, indices_aleatorios_validation)

diferentes_12 = np.setdiff1d(pixels_sem_ruido_train, pixels_sem_ruido_train.astype(np.float64))

if diferentes_12.size > 0:
  print("Há números repetidos entre array1 e array2:", diferentes_12)
else:
  print("Iguais")

#if repetidos_12.size > 0:
#    print("Há números repetidos entre array1 e array2:", repetidos_12)
#if repetidos_13.size > 0:
#    print("Há números repetidos entre array1 e array3:", repetidos_13)
#if repetidos_23.size > 0:
#    print("Há números repetidos entre array2 e array3:", repetidos_23)

#if repetidos_12.size == 0 and repetidos_13.size == 0 and repetidos_23.size == 0:
#    print("Não há números repetidos entre quaisquer dois dos três arrays.")