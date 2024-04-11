import numpy as np

X = np.array([[3, 2, 250], [13, 62, 10]]) # Exemplo de matriz
dimension_x = X.shape # Dimensionamento da matriz
num_elements = dimension_x[0]*dimension_x[1] # Número de elementos na matriz

power_x = np.sum(X**2) / num_elements # Potência da matriz X
peak_power = 255**2 # Potência de pico

desired_psnr = 10 # PSNR desejada em dB

# Cálculo da potência do ruído necessária para atingir a PSNR desejada
noise_power = peak_power / (10**(desired_psnr / 10))

# Geração da matriz de ruído com a potência de ruído desejada
noise_matrix = np.random.normal(0, np.sqrt(noise_power), X.shape)
# Cálculo da potência do ruído
power_noise = np.sum(noise_matrix**2) / num_elements

# Cálculo do SNR e PSNR
snr = 10*np.log10(power_x / power_noise)
psnr = 10*np.log10(peak_power / power_noise)

print("desired_noise_power =", noise_power)
print("obtained power_noise =", power_noise)
print("noise_matrix =", noise_matrix)
print("num_elements =", num_elements)
print("power_x =", power_x, "Watts")
print("power_noise =", power_noise, "Watts")

print("SNR =", snr, "dB")
print("PSNR =", psnr, "dB")