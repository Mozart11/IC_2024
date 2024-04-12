import numpy as np

# Suponha que 'array1', 'array2' e 'array3' sejam seus arrays NumPy
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 8])
array3 = np.array([8, 9, 10, 11, 12])

# Verificar se há números repetidos entre quaisquer dois dos três arrays
repetidos_12 = np.intersect1d(array1, array2)
repetidos_13 = np.intersect1d(array1, array3)
repetidos_23 = np.intersect1d(array2, array3)

if repetidos_12.size > 0:
    print("Há números repetidos entre array1 e array2:", repetidos_12)
if repetidos_13.size > 0:
    print("Há números repetidos entre array1 e array3:", repetidos_13)
if repetidos_23.size > 0:
    print("Há números repetidos entre array2 e array3:", repetidos_23)

if repetidos_12.size == 0 and repetidos_13.size == 0 and repetidos_23.size == 0:
    print("Não há números repetidos entre quaisquer dois dos três arrays.")