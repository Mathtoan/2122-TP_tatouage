import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

dir_figure = './figure'

if not os.path.exists(dir_figure):
    os.makedirs(dir_figure)

# Lecture de l'image
I = plt.imread('cameraman.tif')
I_fft2 = np.fft.fft2(I)
I_fftshift = np.fft.fftshift(I_fft2)

# Generation du motif de tatouage
T = np.random.randn(1024)
alpha = 0.05

def insertion_tattoo(I,T,alpha):
    """
    Input :
        I : image a tatouer
        T : motif du tatouage
        alpha : coefficient d'application du tatouage
    Output :
        I_tattooed : image tatouee
    
    Applique le motif T sur l'image I sur les basses frequences du spectre frequenciel
    avec le coefficient alpha pour generer l'image I_tattooed.

    Rq : dans le cadre de ce TP, cette fonction n'acceptera que des motifs de tatouage
    de type vecteur dont la longueur est un carré d'un entier
    """
    n = int(T.shape[0]**0.5)
    I_tattooed_fft2 = np.fft.fft2(I) # FFT
    I_tattooed_fft2_shift = np.fft.fftshift(I_tattooed_fft2) # FFT shift

    # Application du motif de tatouage sur l'image dans l'espace de Fourrier
    for i in range(n):
        for j in range(n):
            I_tattooed_fft2[i+1][j+1] = I_tattooed_fft2[i+1][j+1] * (1+alpha*T[i*n+j])
            I_tattooed_fft2[-1-i][-1-j] = I_tattooed_fft2[-1-i][-1-j] * (1+alpha*T[i*n+j])

            I_tattooed_fft2[i+1][-1-j] = I_tattooed_fft2[i+1][-1-j] * (1+alpha*T[i*n+j])
            I_tattooed_fft2[-1-i][j+1] = I_tattooed_fft2[-1-i][j+1] * (1+alpha*T[i*n+j])
    
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

    # Passage dans l'espace direct
    I_tattooed = np.fft.ifft2(I_tattooed_fft2)

    # Affichage

    plt.figure()
    plt.subplot(131)
    plt.imshow(I, 'gray')
    plt.title('Image originale')
    plt.subplot(132)
    plt.imshow(np.abs(np.log(I_tattooed_fft2)), 'gray')
    plt.title('TF de l image')
    plt.subplot(133)
    plt.imshow(np.abs(np.log(I_tattooed_fft2_shift)), 'gray')
    plt.title('TF shift de l image')
    plt.savefig(os.path.join(dir_figure, 'image avec fft.png'))
    # plt.show()
    plt.close()

    plt.figure()
    plt.subplot(121)
    plt.imshow(I, 'gray')
    plt.title('Image originale')
    plt.subplot(122)
    plt.imshow(np.abs(I_tattooed), 'gray')
    plt.title('Image tatouee')
    plt.savefig(os.path.join(dir_figure, 'image vs tatoue.png'))
    # plt.show()
    plt.close()

    return I_tattooed

def PSNR(I, I_tattooed):
    """
    Input :
        I : image originale
        I_tattooed : image tatouee
    Output :
        psnr : Peak Signal Noise Ratio entre l'image original et l'image tatouee
        
    Calcul du PSNR entre l'image originale et l'image tatouée
    """

    MSE = 0 # Mean Square Error
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            MSE += (I[i][j]-I_tattooed.real[i][j])**2
    MSE = MSE/(I.shape[0]*I.shape[1])

    psnr = 10*np.log10((np.amax(I)**2)/MSE)

    return psnr

# alphas = np.linspace(0, 1, 21)
# alphas[0] = alphas[0]+0.01
# psnr = []
# for k in alphas:
#     I_tattooed = insertion_tattoo(I,T,k)
#     psnr.append(PSNR(I, I_tattooed))

# plt.figure(dpi=200)
# plt.plot(alphas, psnr)
# plt.grid()
# plt.xlabel('alpha')
# plt.ylabel('PSNR')
# plt.title('Evolution du PSNR en fonction de alpha')
# plt.savefig(os.path.join(dir_figure, 'PSNR.png'))
# plt.close()

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

def detection_tattoo(I, I_tattooed, T, alpha, display=True):
    """
    Input :
        I : image originale
        I_tattooed : image tatouee
        T : motif du tatouage
        alpha : coefficient d'application du tatouage
        display : affichage des tatouages (True par defaut)
    Output :
        detected_tattoo : Indique si le tatouage est detecte
        match_tattoo : Indique si le tatoauge correspond au tatouage original
        
    Detecte la presence ou non du tatouage, et si oui, calcul coefficient de correlation
    pour determiner si le tatouage correspond au tatouage original.
    """

    detected_tattoo = False
    match_tattoo = False

    # FFT sur l'image originale et tatouee
    I_fft2 = np.fft.fft2(I)
    I_tattooed_fft2 = np.fft.fft2(I_tattooed)

    # Recuperation du tatouage sur l'image tatouee
    T_est = ((I_tattooed_fft2/I_fft2) - 1)/alpha

    # Generation du tatouage a partir du motif
    n = int(T.shape[0]**0.5)
    T_real = np.zeros(I.shape)

    for i in range(n):
        for j in range(n):
            T_real[i+1][j+1] += T[i*n+j]
            T_real[-1-i][-1-j] += T[i*n+j]

            T_real[i+1][-1-j] += T[i*n+j]
            T_real[-1-i][j+1] += T[i*n+j]

    # Affichage du tatouage original et du tatouage detectee
    if display:
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.abs(T_est), 'gray')
        plt.title('Tatouage estime')
        plt.subplot(122)
        plt.imshow(np.abs(T_real), 'gray')
        plt.title('Tatouage reel')
        plt.savefig(os.path.join(dir_figure, 'tatouage estime vs reel.png'))
        # plt.show()
        plt.close()

    # Vectorisation des tatouage, utile pour les calculs
    N = I.shape[0] * I.shape[0]
    T_est_vect = np.reshape(T_est, N)
    T_real_vect = np.reshape(T_real, N)

    # Detection du tatouage
    T_est_norm = np.linalg.norm(T_est_vect)
    print(T_est_norm)
    if T_est_norm < 1 :
        print('Aucun tatouage dectecte')
        return detected_tattoo, match_tattoo
    else:
        detected_tattoo = True
        print('Tatouage detecte')

        # Calcul du coefficient de correlation
        mean_T_est = np.mean(T_est_vect)
        mean_T_real = np.mean(T_real_vect)
        threshold = .5
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(N):
            sum1 += (T_est_vect[i] - mean_T_est)*(T_real_vect[i]-mean_T_real)
            sum2 += (T_est_vect[i] - mean_T_est)**2
            sum3 += (T_real_vect[i] - mean_T_real)**2
        
        gamma = sum1/np.sqrt(sum2*sum3)
        print('Coefficient de correlation :', gamma.real)

        if gamma<threshold :
            print('Le tatouage detecte ne correspond pas au tatouage de base')
            return detected_tattoo, match_tattoo
        else:
            match_tattoo = True
            print('Le tatouage detecte correspond au tatouage de base')
            return detected_tattoo, match_tattoo

# Detection de tatouage
alpha = 0.1
I_tattooed = insertion_tattoo(I, T, alpha)
print(detection_tattoo(I, I_tattooed, T, alpha, display=False))

# Degradation du tatouage

gaussian_noise = np.random.normal(0, 0.001 ** 0.5, I_tattooed.shape)*255


I_tattooed_gauss = I_tattooed.real+gaussian_noise

print(np.amax(I_tattooed.real), np.amin(I_tattooed.real))

plt.figure()
plt.subplot(121)
plt.imshow(I_tattooed.real, 'gray')
plt.title('Image tatouee')
plt.subplot(122)
plt.imshow(I_tattooed_gauss.real, 'gray')
plt.title('Image degradee')
plt.show()
plt.close()

print(detection_tattoo(I, I_tattooed_gauss, T, alpha))