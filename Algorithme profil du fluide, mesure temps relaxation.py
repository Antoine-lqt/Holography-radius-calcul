# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:07:29 2022

@author: loqueta
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import erosion
from skimage.morphology import disk  # noqa
import cv2
import os.path
from fonction import normalisation, lissage, sommelignes
from scipy import interpolate


# =============================================================================
# Tri des images.tiff 
# Sorting images.tiff 
# =============================================================================
plt.close()
repertoire=[]
repertoire=os.listdir("File path" )


fichiers=[]
#Sélection que des holos 
for i in repertoire :
    if '.tif' in i:
        fichiers.append(i) 

# =============================================================================
# Début Algoritme temps de relaxation 
# Start Relaxation time algorithm
# =============================================================================
rayon=[]
temps=[]
x0=np.linspace(0,4096,4096)


# =============================================================================
# 1 Calcul du rayon  
# 1 Radius calculation  
# =============================================================================
# =============================================================================
# 1.1 Détection filament 
# 1.1 Filament detection 
# =============================================================================
print("Combien d'image du pont liquide ?")
plq=int(input())
m=r=plq
while r < len(fichiers) :  

    holo=np.mean(Image.open(r'File path'+fichiers[r]),-1)
    holo=np.array(holo)
    
    footprint = disk(6)
    eroded_holo = erosion(holo, footprint) # Grossisment des bords par érosion
    
    
    
    tableau= normalisation(eroded_holo)

    
    shape = np.shape(tableau)
    maximum=[] # Liste récupérant profil de droite de la structure liquide
    minimum=[] # Liste récupérant profil de gauche de la structure liquide 
    #Algo détermination profil G et D du fluide 
    for i in range(0,shape[0]): # Parcours lignes matrice
        profil=normalisation(tableau[i,:]) # Création profil d'une ligne 
        temp=[]  # Création liste temporaire 

        for j in range(0,shape[1]):  # Parcours colonnes du profil 
            if profil[j]<100 :  # Critère 100 "empirique" 
                temp.append(profil[j])
                
            if len(temp)>=2 and j==shape[1]-1:  # Test à la fin de la ligne la longeur 
            
                temp_max=[]
                temp_min=[]
                #Sépare min et max lorsque profil à la même valeur à la mi-hauteur :
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
        
    

# =============================================================================
# 1.2 Lissage 
# 1.2 Smoothing 
# =============================================================================  
    lx,ly=lissage(x0,fil,100) # Moyenne les valeurs par pas de 100 pixels 
                
    counts_r,vals=np.histogram(ly,bins=int(np.max(lx)),range=(np.min(ly),np.max(ly)))
                
    zone_fil=np.argwhere(ly<vals[np.argwhere(counts_r[0:1000]>15)[-1]])
                
    p1=zone_fil[0][0]# Segment sup du filament 
    p2=zone_fil[-1][0]   # Segment inf du filament 
   
    plt.figure(),plt.imshow(holo<100),plt.plot(maximum,x0), plt.plot(minimum,x0),plt.axhline(p1),plt.axhline(p2)

# =============================================================================
# 1.3 Calcul rayon filament sur la segmentation
# 1.3 Filament radius calculation on segmentation  
# =============================================================================
    Isb_resize=holo[p1:p2,:] # Resize hologramme
                
    Isb_resize_norm=normalisation(Isb_resize)
             
                

    Isb_somme=np.mean(Isb_resize_norm,axis=0)
    Isb_somme_norm = normalisation(Isb_somme)


    counts,vals=np.histogram(Isb_somme_norm,bins=int(np.max(Isb_somme_norm)),range=(np.min(Isb_somme_norm),np.max(Isb_somme_norm)))
    
    valeur_haute=vals[np.argmax(counts)]
    valeur_basse=np.min(Isb_somme_norm)
    y_taille_fila=(valeur_haute-valeur_basse)/2 

#Interpolation des points pour 
    f = interpolate.interp1d(x0, Isb_somme_norm, fill_value='extrapolate')
    xint=np.linspace(0,4096,4096*100)
    profil_extra=f(xint)



# Déterminer intersection entre y_taille_fila et profil_extra
    taille_fila=np.argwhere(profil_extra<=y_taille_fila)  #Condition lorsque courbe au dessus de l'autre + indexation
    x1=int(taille_fila[0])/100   #Valeur minimale 
    x2=int(taille_fila[-1])/100  #Valeur maximale 

    p = 5.5e-3
    new_rayon=((x2-x1)*p)/2

    if len(rayon)>=3:
        if rayon[-2]<=rayon[-1]<=new_rayon:
            print("arret",r-plq)
            r=len(fichiers)
        else :
            rayon.append(new_rayon)
            print("suivant2",r-plq)
            r+=1
    if len(rayon)<3: 
        rayon.append(new_rayon)
        print("suivant1",r-plq)
        r+=1
   
# =============================================================================
# 2. détermination temps 
# 2. time determination 
# =============================================================================

m=plq

temps=[]
for m in range(plq,len(rayon)+plq):
    if m == plq :
        nom=str(fichiers[m])
        date=nom[:12]
        heures = date[:2]
        minutes=(int(date[3:5]) + int(heures))*60
        secondes=float(date[6:]) + minutes 
        t0= secondes
        temps.append(t0-t0)
        m+=1
    else : 
        nom=str(fichiers[m])
        date=nom[:12]
        heures = date[:2]
        minutes=(int(date[3:5]) + int(heures))*60
        secondes=float(date[6:]) + minutes 
        t=secondes 
        temps.append(t-t0)
        m+=1

rayon=np.array(rayon)
temps=np.array(temps)

# The relaxation time (tho) is given by: radius= exp(-temps/(3*tho))

# To compare curve fitting with the literature, I take the values for radius and time, then plot them in Excel, but it's perfectly possible to do it in python with scipy.optimize.curve_fit 

# Le temps de relaxation (tho) est donné par : rayon= exp(-temps/(3*tho))

# Afin de comparer le fitting des courbes à la littérature, je prends les valeurs du rayon et du temps, puis je trace sur Excel, mais c'est tout à fait possible de le faire par python avec scipy.optimize.curve_fit 


