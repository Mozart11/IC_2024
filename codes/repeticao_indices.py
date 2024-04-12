# Verificação de repetição dos indices
# Verificar se há números repetidos entre quaisquer dois dos três arrays
repetidos_12 = np.intersect1d(indices_aleatorios_teste, indices_aleatorios_validation)
repetidos_13 = np.intersect1d(indices_aleatorios_teste, indices_aleatorios_train)
repetidos_23 = np.intersect1d(indices_aleatorios_train, indices_aleatorios_validation)

if repetidos_12.size > 0:
    print("Há números repetidos entre array1 e array2:", repetidos_12)
if repetidos_13.size > 0:
    print("Há números repetidos entre array1 e array3:", repetidos_13)
if repetidos_23.size > 0:
    print("Há números repetidos entre array2 e array3:", repetidos_23)

if repetidos_12.size == 0 and repetidos_13.size == 0 and repetidos_23.size == 0:
    print("Não há números repetidos entre quaisquer dois dos três arrays.")