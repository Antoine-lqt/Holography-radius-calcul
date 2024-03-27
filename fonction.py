

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import erosion
from skimage.morphology import disk  # noqa
import cv2 
import os.path



p = 5.5e-6  # taille pixel en mètres
jmax, imax = 4096, 4096
j, i = np.mgrid[0:jmax, 0:imax]

x = (i-imax/2)*p
y = (j-jmax/2)*p


L = 520e-9
im = complex(0, 1)
zropt_max = 0.44819999999998916

def comparaison(liste, seuil, tolérance):
    i=0
    for i in range(0,len(liste)):
        if seuil[i]-int(tolérance) < liste[i]<  seuil[i]+ int(tolérance):
            pass
        else : 
            print("Rupture",i)
       
        i=i+1

def conversion(image):
    j=0
    ma=int(np.max(image))
    mi=int(np.min(image))
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i][j]> (ma+mi)/2:
                image[i][j]=1 
            else :
                image[i][j]=0
            j=j+1
        i=i+1
    return image 

def conversion1colonne(image):
    ma=int(np.max(image))
    mi=int(np.min(image))
    for i in range(0, len(image)):
        if image[i]> (ma+mi)/2:
            image[i]=1 
        else :
            image[i]=0

        i=i+1
        return image
    

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def lissage(Lx,Ly,p):
    '''Fonction qui débruite une courbe par une moyenne glissante
    sur 2P+1 points'''
    Lxout=[]
    Lyout=[]
    for i in range(p,len(Lx)-p):   
        Lxout.append(Lx[i])
    for i in range(p,len(Ly)-p):
        val=0
        for k in range(2*p):
            val+=Ly[i-p+k]
        Lyout.append(val/2/p)
            
    return Lxout,Lyout

def most_frequent(List): 
    return max(set(List), key = List.count) 



import numba as nb
@nb.njit
def normalisation(fonction):
    return 255*((fonction-fonction.min())/(fonction.max()-fonction.min()))
@nb.njit
def profil(eroded_holo):
    x0=np.linspace(0,4096,4096)
    tableau= normalisation(eroded_holo)

    
    shape = np.shape(tableau)
    maximum=[] # Liste récupérant profil de droite de la structure liquide
    minimum=[] # Liste récupérant profil de gauche de la structure liquide 
    #Algo détermination profil G et D du fluide 
    for i in range(0,shape[0]): # Parcours lignes matrice
        profil=normalisation(tableau[i,:]) # Création profil d'une ligne 
        temp=[]  # Création liste temporaire 

        for j in range(0,shape[1]):  # Parcours colonnes du profil 
            if profil[j]<100 :  # Critère 70 "empirique" 
                temp.append(profil[j])
                
            if len(temp)>=2 and j==shape[1]-1:  # Test à la fin de la ligne la longeur 
                # maximum.append(np.argwhere(profil==temp[-1])[0][0])  
                # minimum.append(np.argwhere(profil==temp[0])[0][0])
                temp_max=[]
                temp_min=[]
                
                for g in range(0,len(profil)):
                    if profil[g]==temp[-1]:
                        temp_max.append(g)
                        g+=1
                    else : 
                        g+=1
                        pass 
                for g in range(0,len(profil)):
                    if profil[g]==temp[0]:
                        temp_min.append(g)
                        g+=1
                    else : 
                        g+=1
                        pass 
                    
                maximum.append(temp_max[-1])
                minimum.append(temp_min[0])
            if len(temp)<2 and j==shape[1]-1:
                while len(temp)<2 and j==shape[1]-1: 
                    temp=[]
                    m=1  # Critère que l'on fait augmenter 
                    for j2 in range(0,shape[1]):
                        if profil[j2]<100+m :
                            temp.append(profil[j2])
                        else:
                           m+=1
                        
                # maximum.append(np.argwhere(profil==temp[-1])[0][0])
                # minimum.append(np.argwhere(profil==temp[0])[0][0])
                temp_max=[]
                temp_min=[]
                
                for g in range(0,len(profil)):
                    if profil[g]==temp[-1]:
                        temp_max.append(g)
                        g+=1
                    else : 
                        g+=1
                        pass 
                for g in range(0,len(profil)):
                    if profil[g]==temp[0]:
                        temp_min.append(g)
                        g+=1
                    else : 
                        g+=1
                        pass 
                    
                maximum.append(temp_max[-1])
                minimum.append(temp_min[0])
           
        fil = np.array(maximum) - np.array(minimum) # Profil du fluide
       
        
    return fil 


def resize(profil, holo, fin):
    counts, vals = np.histogram(profil, range=(
        np.min(profil), np.max(profil)), bins=int(np.max(profil)))

    valeur_haute = vals[np.argmax(counts)]
    valeur_basse = np.min(profil)
    y_taille_fila = (abs(valeur_haute)-abs(valeur_basse))/2

    # Déterminer intersection entre y_taille_fila et test_lignes2_extra
    # Condition lorsque courbe au dessus de l'autre + indexation
    taille_fila = np.argwhere(y_taille_fila >= profil)
    x1 = int(taille_fila[0])  # Valeur minimale
    x2 = int(taille_fila[-1])  # Valeur maximale

    holo = holo[0:4096, x1-25:x2+25]
    fin = fin[0:4096, x1-25:x2+25]
    return holo, fin

def restitution(holographie):
    h = (1/(im*zropt_max*L))*np.exp(im*np.pi *
                                    (x**2+y**2)/(L*zropt_max))  # Noyau de Fresnel
    H = np.fft.fft2(h)  # Transformée de Fourrier du noyau
    T = np.fft.fft2(holographie)   # Transfromée de Fourrier de mon holo

    PC = T*H     # Produit de convolution
    pc = np.fft.ifft2(PC)  # Transofrmée de Fourrier inverse du PC
    pc = np.fft.fftshift(pc)  # Recentrée

    holographie_restituee = abs(pc)  # Holo reconstruite
    return holographie_restituee


def sommecolonnes(tableau):
    return np.sum(tableau, axis=1)

def sommelignes(tableau):
    return np.sum(tableau, axis=0)
