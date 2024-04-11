import matplotlib.pyplot as plt
import numpy as np
import cv2
 
with open("../datasets/t10k-images.idx3-ubyte", "rb") as f:  # Imagens do conjunto de teste MNIST
   magic_number = int.from_bytes(f.read(4), 'big')  # Metadados
   num_images = int.from_bytes(f.read(4), 'big') 
   num_rows = int.from_bytes(f.read(4), 'big') # 28
   num_cols = int.from_bytes(f.read(4), 'big') # 28 > 1 imagem
   
   hexs_b = (f.read()) # Retorna uma lista de binarios
   pixels = np.frombuffer(hexs_b, dtype=np.uint8) # Transforma a sequencia de bits conforme o dtype, 
   pixels_reshape = pixels.reshape(num_images, num_rows, num_cols) # Unsigned_8 = Pixel Format = 0 a 255 valores

   plt.imshow(pixels_reshape[0], cmap='gray')  # plt para plotar imagem
   # plt.show()


# img = cv2.imread("maca.jpg",0) # << ler em byte
img = pixels_reshape[0]
img = img / 255  # Porque ? > Normaliza os valores entre 0 e 1
x,y = img.shape
psnr = float(input("PSNR: "))

def gen_gaussia_noise(var,x,y):
    sigma = np.sqrt(var)
    n = np.random.normal(loc=0,
                         scale=sigma,
                         size=(x,y))
    return n

def add_noise(noise,image):
    # corrupt image = original image + gaussian noise
    corrupt = image + noise
    return corrupt

def find_variance(img,target_psnr):
    # sigma^2 == MSE
    sigma = np.amax(img)/ np.power(10,target_psnr/10)
    return sigma

sigma = find_variance(img,psnr)
noise = gen_gaussia_noise(sigma,x,y)
corrupt_img = add_noise(noise,img)

cv2.imshow("original image",img)
cv2.imshow("gaussian noise",noise)   # Porque preto e branco ?
cv2.imshow("corrupted image",corrupt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()