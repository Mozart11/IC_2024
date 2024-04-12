import matplotlib.pyplot as plt
import numpy as np
import cv2
 
def openDatasets(path, c):  # path e tipo de conjunto (train,val ou test)
    with open("../datasets/" + path, "rb") as f:
        bins = f.read()
        pixels = np.frombuffer(bins, dtype=np.uint8)
        pixels_reshape = pixels.reshape(c, 28, 28)
        return(pixels_reshape)

test_validation_shape0 = 14000
train_shape0 = 42000

pixels = openDatasets("train.bin", train_shape0)

# img = cv2.imread("maca.jpg",0) # << ler em byte
img = pixels[7]
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

#corrupt_img = corrupt_img.astype(np.uint8)

cv2.imshow("original image",img)
cv2.imshow("gaussian noise",noise)   # Porque preto e branco ?
cv2.imshow("corrupted image",corrupt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()