import numpy as np
import cv2

# the code only suports Black and White images, you can use colorized photos
# img = cv2.imread("maca.jpg",0) # << ler em byte
img = img / 255
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
cv2.imshow("gaussian noise",noise)
cv2.imshow("corrupted image",corrupt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()