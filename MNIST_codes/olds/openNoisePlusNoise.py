import numpy as np
import cv2
import matplotlib.pyplot as plt
from codes.sets_gen_noise import find_variance, gen_gaussia_noise, add_noise, openDatasets, openDatasets_f

np.random.seed(333)

# Script para testar conjuntos + adicionar ruido again 

test_validation_shape0 = 14000
train_shape0 = 42000

pixels_com_ruido_train = openDatasets_f("train_noise.bin", train_shape0)
pixels_com_ruido_test = openDatasets_f("test_noise.bin", test_validation_shape0)
pixels_com_ruido_validation = openDatasets_f("validation_noise.bin", test_validation_shape0)

pixels_sem_ruido_train = openDatasets("train.bin", train_shape0)

#img = cv2.imread("../maca.jpg",0) # << ler em byte
img = pixels_com_ruido_train[0]
#img = img / 255
x,y = img.shape
psnr = float(input("PSNR: "))

sigma = find_variance(img,psnr)
noise = gen_gaussia_noise(sigma,x,y)
corrupt_img = add_noise(noise,img)

plt.imshow(corrupt_img, cmap="gray")
#plt.show()

cv2.imshow("original image",img)
cv2.imshow("gaussian noise",noise)
cv2.imshow("corrupted image",corrupt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()