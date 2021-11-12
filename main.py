import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

I = plt.imread('cameraman.tif')
I_fft2 = np.fft.fft2(I)
I_fftshift = np.fft.fftshift(I_fft2)


T = np.random.randn(1024)
alpha = 0.05

def insertion_tattoo(I,T,alpha):
    I_tattooed_fft2 = np.fft.fft2(I) # FFT

    for i in range(32):
        for j in range(32):
            I_tattooed_fft2[i+1][j+1] = I_tattooed_fft2[i+1][j+1] * (1+alpha*T[i*32+j])
            I_tattooed_fft2[-1-i][-1-j] = I_tattooed_fft2[-1-i][-1-j] * (1+alpha*T[i*32+j])

            I_tattooed_fft2[i+1][-1-j] = I_tattooed_fft2[i+1][-1-j] * (1+alpha*T[i*32+j])
            I_tattooed_fft2[-1-i][j+1] = I_tattooed_fft2[-1-i][j+1] * (1+alpha*T[i*32+j])
    
    # # Verification symetrie hermitienne (on verifie ligne par ligne)
    # for i in range(1, int(256/2)):
    #     sum = 0
    #     for j in range(1, int(256/2)):
    #         sum += np.abs(I_tattooed_fft2[i][j]) - np.abs(I_tattooed_fft2[-i][-j])
    #     print(sum)
    
    # for i in range(1, int(256/2)):
    #     sum = 0
    #     for j in range(1, int(256/2)):
    #         sum += np.abs(I_tattooed_fft2[i][-j]) - np.abs(I_tattooed_fft2[-i][j])
    #     print(sum)

    I_tattooed = np.fft.ifft2(I_tattooed_fft2)
    return I_tattooed

def PSNR(I, I_tattooed):
    MSE = 0
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            MSE += (I[i][j]-I_tattooed.real[i][j])**2
    MSE = MSE/(I.shape[0]*I.shape[1])

    return 10*np.log10((np.amax(I)**2)/MSE)

alpha = np.linspace(0, 1, 21)
psnr = []
for k in alpha:
    I_tattooed = insertion_tattoo(I,T,k)
    psnr.append(PSNR(I, I_tattooed))

plt.figure()
plt.plot(alpha, psnr)
plt.grid()
plt.xlabel('alpha')
plt.ylabel('PSNR')
plt.title('Evolution du PSNR en fonction de alpha')
plt.show()

# plt.figure()
# plt.subplot(131)
# plt.imshow(I, 'gray')
# plt.title('Image')
# plt.subplot(132)
# plt.imshow(np.log(np.abs(I_fft2)), 'gray')
# plt.title('FFT2')
# plt.subplot(133)
# plt.imshow(I_tattooed.real, 'gray')
# plt.title('FFT shift')
# plt.show()