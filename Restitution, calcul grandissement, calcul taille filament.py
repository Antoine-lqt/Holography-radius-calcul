# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:46:01 2022

@author: loqueta
"""

import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 

p=5.5e-6 # taille pixel en mètres
jmax, imax = 4096, 4096 # Taille de la caméra 
jp, ip = np.mgrid[0:jmax,0:imax]
A=np.zeros((4096,4096))

x=(ip-imax/2)*p
y=(jp-jmax/2)*p

holo = Image.open(r'File path') # Hologramme a restitué 
holo = np.array(holo) # Conversion image en tableau 

fond = Image.open(r'File path') # Hologramme de la mire seule 
Fond = np.array(fond) 

L=520e-9 #Lambda en mètre 
im=complex(0,1)

# =============================================================================
# 1. Restitution de l'hologramme
# =============================================================================
Isb=holo/Fond # Supression du fond et de la mire  

plt.figure('Holograme sans fond')
plt.title('Holograme sans fond')
plt.legend(title="Cliquer sur filament")
plt.plot(x,y)
plt.xlabel('i (pixels)')
plt.ylabel('j (pixels)')
plt.imshow(Isb,cmap=plt.get_cmap('Greys_r'))

# =============================================================================
# 1.a. Selectionne sur l'hologramme ( 1 point ) 
# =============================================================================
XY=plt.ginput(1)
XY=np.round(XY,0)
J1=int(XY[0][0])
I1=int(XY[0][1])

plt.close()

# =============================================================================
# 1.b. Redimension de la taille du noyau de Fresnel et de l'hologramme
# =============================================================================
Isb_resize=Isb[I1:I1+100,:] # Resize hologramme

new_jmax, new_imax = len(Isb_resize[:]), len(Isb_resize[0,:]) # Resize noyau de Fresnel
new_j, new_i = np.mgrid[0:new_jmax,0:new_imax]

# =============================================================================
# 1.c. Algorithme de détermination de Zropt 
# =============================================================================

print('Donner la valeur minimale du balayage en mètre')
binf=float(input())
print('Donner la valeur maximale du balayage en mètre')
bmax=float(input())
print('Donner le pas du balayage')
bpas=float(input())

c=[]
z=np.arange(binf,bmax,bpas)

for k in z:
    x1=(new_i-new_imax/2)*p
    y1=(new_j-new_jmax/2)*p
    h=(1/(im*k*L))*np.exp(im*np.pi*(x1**2+y1**2)/(L*k)) #noyau de fresnel
    H=np.fft.fft2(h)   #Transformée de Fourrier du noyau
    T=np.fft.fft2(Isb_resize)   # Transfromée de Fourrier de l'hologramme redimensionné
    PC=T*H     # Produit de convolution 
    pc=np.fft.ifft2(PC)    #Transformée de Fourrier inverse du PC 
    oui=np.fft.fftshift(pc)
 
    #Calcul critère 
    Re=np.min(oui.real) # On peut utiliser le rapport des variances mais les contrastes ne sont pas assez imporants sur cette série 
    c.append(Re)
    
zropt_max = z[np.argmin(c)]
print('Zr optimale est',zropt_max,'m')

# =============================================================================
# 1.d. Restitution au Zropt : 
# =============================================================================
h= (1/(im*zropt_max*L))*np.exp(im*np.pi*(x**2+y**2)/(L*zropt_max)) # Noyau de Fresnel 
H=np.fft.fft2(h)   #Transformée de Fourrier du noyau 
T=np.fft.fft2(Isb)   # Transfromée de Fourrier de mon holo 

PC=T*H     # Produit de convolution 
pc=np.fft.ifft2(PC)    #Transofrmée de Fourrier inverse du PC 
pc=np.fft.fftshift(pc) # Recentrée 

Aopt=abs(pc) # Holo reconstruite 

# =============================================================================
# 1.e. Affichage avant / après traitement d'image + Critère + Critère zoommé
# =============================================================================
fig = plt.figure(figsize=(12,10)) # Taille de la figure 
  
rows = 2
columns = 2
  
fig.add_subplot(rows, columns, 1) # Affiche l'hologramme orginal
plt.title('Hologramme original') 
plt.xlabel('i pixels')
plt.ylabel('j pixels')
plt.imshow(holo, cmap=plt.cm.gray) 

  
fig.add_subplot(rows, columns, 2)  # Affiche l'hologramme sans fond restitué
plt.title('Hologramme sans fond resituté') 
plt.xlabel('i pixels')
plt.ylabel('j pixels')
plt.imshow(Aopt, cmap=plt.cm.gray) 

  
fig.add_subplot(rows, columns, 3)   # Affiche le critère de sélection
plt.plot(zropt_max,min(c), marker=".", color="red")
plt.plot(z,c)
plt.xlabel('z')
plt.ylabel('c')
plt.title("Critère") 
plt.show()
  

fig.add_subplot(rows, columns, 4) # Affiche le zoom sur le critère  
plt.plot(zropt_max,min(c), marker=".", color="red")
b1=(zropt_max,min(c))
plt.plot(z,c)
plt.xlabel('z')
plt.ylabel('c')
plt.xticks(rotation='45')
plt.title("Critère zoommé") 
plt.xlim(zropt_max-10*bpas,zropt_max+10*bpas)
plt.ylim(min(c)-1e13*bpas,min(c)+1e13*bpas)
plt.show()

# =============================================================================
# 2. Calcul des grandissements
# =============================================================================
# =============================================================================
# 2.1 Calcul du grandissement horizontal 
# =============================================================================
# =============================================================================
# 2.1.a. Selection 2x le centre de la mire 
# =============================================================================
plt.figure()
plt.title('Selectionner le centre de la mire')
plt.imshow(Fond)

XY=plt.ginput(1)
XY=np.round(XY,0)
J1=int(XY[0][0])
I1=int(XY[0][1])

plt.close()
Fond_resize=Fond[I1-512:I1+512,J1-512:J1+512]

plt.figure()
plt.title('Selectionner le centre de la mire')
plt.imshow(Fond_resize)

XY=plt.ginput(1)
XY=np.round(XY,0)
J2=int(XY[0][0])
I2=int(XY[0][1])
plt.close()
Fond_resize2=Fond_resize[I2-256:I2+256,J2-256:J2+256]

# =============================================================================
# 2.1.b. Restituion de la mire au Zropt 
# =============================================================================

c2=[]
z2=np.arange(0.001,0.04,0.0001)
new_jmax2, new_imax2 = len(Fond_resize2[:]), len(Fond_resize2[0,:]) # Resize noyau de Fresnel
new_j2, new_i2 = np.mgrid[0:new_jmax2,0:new_imax2]


for k in z2:
    x2=(new_i2-new_imax2/2)*p
    y2=(new_j2-new_jmax2/2)*p
    h=(1/(im*k*L))*np.exp(im*np.pi*(x2**2+y2**2)/(L*k)) #noyau de fresnel
    H=np.fft.fft2(h)   #Transformée de Fourrier du noyau
    T=np.fft.fft2(Fond_resize2)   # Transfromée de Fourrier de mon holo resize
    PC=T*H     # Produit de convolution 
    pc=np.fft.ifft2(PC)    #Transformée de Fourrier inverse du PC 
    oui=np.fft.fftshift(pc)
 
   
    #Calcul critère 
    Re=np.var(oui.real)
    Im=np.var(oui.imag)
    c2.append(Re/Im)

zropt_max2 = z2[np.argmax(c2)]

h= (1/(im*zropt_max2*L))*np.exp(im*np.pi*(x2**2+y2**2)/(L*zropt_max2)) # Noyau de Fresnel 
H=np.fft.fft2(h)   #Transformée de Fourrier du noyau 
T=np.fft.fft2(Fond_resize2)   # Transfromée de Fourrier de la mire 

PC=T*H     # Produit de convolution 
pc=np.fft.ifft2(PC)    #Transofrmée de Fourrier inverse du PC 
pc=np.fft.fftshift(pc) # Recentrée 


mire_h_restitu=abs(pc)

mire_h_restitu_norm=255*((mire_h_restitu-mire_h_restitu.min())
                         /(mire_h_restitu.max()-mire_h_restitu.min())) # Normalisation

f,ax= plt.subplots()
ax.set_xlabel('i (pixels)')
ax.set_ylabel('j (pixels)')
plt.title('Mire restituée')
plt.legend(title='Selectionner le centre du disque central')
plt.imshow(mire_h_restitu_norm,cmap=plt.get_cmap('Greys_r'))
plt.colorbar()

# =============================================================================
# 2.1.c. Taille des disques
# =============================================================================
XY=plt.ginput(1)
XY=np.round(XY,0)
J1=int(XY[0][0])
I1=int(XY[0][1])

plt.close()

mire_h_resize=mire_h_restitu_norm[I1-2:I1+2,:]

ydata_h=np.sum(mire_h_resize,axis=0) #Profil des lignes 

ydata_h_norm=255*((ydata_h-ydata_h.min())/(ydata_h.max()-ydata_h.min()))

#Histogramme du profil : 
counts,vals=np.histogram(ydata_h_norm,bins=int(np.max(ydata_h_norm)),range=(np.min(ydata_h_norm),np.max(ydata_h_norm)))

valeur_haute_h=vals[np.argmax(counts)]
valeur_basse_h=np.min(ydata_h_norm)
y_taille_fila_h=(valeur_haute_h-valeur_basse_h)/2 # y correspondant à la taille du filament 

f,ax= plt.subplots()
ax.set_xlabel('i (pixels)')
ax.set_ylabel('j (pixels)')
plt.title('Profil des lignes')
plt.axhline(y_taille_fila_h,c='r',label='Mi-hauteur')
plt.ylim(-10,260)
plt.plot(ydata_h_norm)

# =============================================================================
# 2.1.d. Interpollation
# =============================================================================#Interpolation
from scipy import interpolate

#Ajout des points pour que le profil croise au mieux la valeur du y trouvée précédement 

xint=(np.linspace(0,512,512*5))
xdata=np.linspace(0,512,512)

f = interpolate.interp1d(xdata, ydata_h_norm, fill_value='extrapolate')
profil_extra=f(xint)
plt.plot(x, y, xint, profil_extra,'xb')
plt.axhline(y_taille_fila_h,c='r')

taille_fila_h=list(np.argwhere(y_taille_fila_h>=ydata_h_norm))

# =============================================================================
# 2.1.e. Separation des 3 pics 
# =============================================================================
liste1_h=[]
liste2_h=[]
liste3_h=[]
mini=min(taille_fila_h)

for k in taille_fila_h :
    if k == mini:
        liste1_h.append(k)
        mini=mini+1

dliste2=np.argmax(liste1_h)
del taille_fila_h[0:dliste2+1]

mini=min(taille_fila_h)
for k in taille_fila_h :
    if k == mini:
        liste2_h.append(k)
        mini=mini+1

dliste3=np.argmax(liste2_h)
del taille_fila_h[0:dliste3+1]

mini=min(taille_fila_h)
for k in taille_fila_h :
    if k == mini:
        liste3_h.append(k)
        mini=mini+1
        
# =============================================================================
# 2.1.f. Calcul grandissement horizontal
# =============================================================================
centre_point1_h=(int(max(liste1_h))-int(min(liste1_h)))/2+int(min(liste1_h))
centre_point2_h=(int(max(liste2_h))-int(min(liste2_h)))/2+int(min(liste2_h))
centre_point3_h=(int(max(liste3_h))-int(min(liste3_h)))/2+int(min(liste3_h))

d12_h=(centre_point2_h-centre_point1_h)*p
d13_h=(centre_point3_h-centre_point1_h)*p
d23_h=(centre_point3_h-centre_point2_h)*p

d12theo_h=d23theo_h=150e-6
d13theo_h=300e-6

K12_h=d12_h/d12theo_h
K13_h=d13_h/d13theo_h
K23_h=d23_h/d23theo_h

K_h=(K12_h+K13_h+K23_h)/3
print('Le grandissement est horizontal',K_h)

# =============================================================================
# 2.2. Calcul grandissement vertical
# =============================================================================
# =============================================================================
# 2.2.a. Rotation 90°
# =============================================================================

rows = len(mire_h_restitu_norm)
cols = len(mire_h_restitu_norm[0])

holo_resize2_ro = [[""] * rows for _ in range(cols)]

for x in range(rows):
    for y in range(cols):
        holo_resize2_ro[y][rows - x - 1] = mire_h_restitu_norm[x][y]
        

mire_v_restitu_norm=np.array(holo_resize2_ro)
plt.figure()
plt.imshow(mire_v_restitu_norm,cmap=plt.get_cmap('Greys_r'))

# =============================================================================
# Selection disque
# =============================================================================
XY=plt.ginput(1)
XY=np.round(XY,0)
J1=int(XY[0][0])
I1=int(XY[0][1])

plt.close()

mire_v_resize=mire_v_restitu_norm[I1-2:I1+2,:]

ydata_v=np.sum(mire_v_resize,axis=0) #Profil des lignes 

ydata_v_norm=255*((ydata_v-ydata_v.min())/(ydata_v.max()-ydata_v.min()))
plt.figure()
plt.plot(ydata_v_norm)

#valeur haute : 
counts_r,vals=np.histogram(ydata_v_norm,bins=int(np.max(ydata_v_norm)),
                           range=(np.min(ydata_v_norm),np.max(ydata_v_norm)))

valeur_haute_r=vals[np.argmax(counts_r)]
valeur_basse_r=np.min(ydata_v_norm)
y_taille_fila_r=(abs(valeur_haute_r)-abs(valeur_basse_r))/2 

# =============================================================================
# Interpollation
# =============================================================================
from scipy import interpolate

xint=(np.linspace(0,512,512*5))
xdata=np.linspace(0,512,512)

f = interpolate.interp1d(xdata, ydata_v_norm, fill_value='extrapolate')
profil_extra_r=f(xint)
axes = plt.gca()
axes.set_ylim(-10, 260)
plt.plot(x, y, xint, profil_extra_r,'xb')
plt.axhline(y_taille_fila_r,c='r')

taille_fila_r=list(np.argwhere(y_taille_fila_r>=ydata_v_norm))

# =============================================================================
# Separation en 3 listes 
# =============================================================================
liste1_r=[]
liste2_r=[]
liste3_r=[]
mini=min(taille_fila_r)

for k in taille_fila_r :
    if k == mini:
        liste1_r.append(k)
        mini=mini+1

dliste2_r=np.argmax(liste1_r)
del taille_fila_r[0:dliste2_r+1]

mini=min(taille_fila_r)
for k in taille_fila_r :
    if k == mini:
        liste2_r.append(k)
        mini=mini+1

dliste3_r=np.argmax(liste2_r)
del taille_fila_r[0:dliste3_r+1]

mini=min(taille_fila_r)
for k in taille_fila_r :
    if k == mini:
        liste3_r.append(k)
        mini=mini+1
        
# =============================================================================
# Calcul grandissement vertical
# =============================================================================
centre_point1_r=(int(max(liste1_r))-int(min(liste1_r)))/2+int(min(liste1_r))
centre_point2_r=(int(max(liste2_r))-int(min(liste2_r)))/2+int(min(liste2_r))
centre_point3_r=(int(max(liste3_r))-int(min(liste3_r)))/2+int(min(liste3_r))

d12_r=(centre_point2_r-centre_point1_r)*p
d13_r=(centre_point3_r-centre_point1_r)*p
d23_r=(centre_point3_r-centre_point2_r)*p

d12theo_r=d23theo_r=250e-6
d13theo_r=500e-6

K12_r=d12_r/d12theo_r
K13_r=d13_r/d13theo_r
K23_r=d23_r/d23theo_r

K_r=(K12_r+K13_r+K23_r)/3
print('Le grandissement est vertical',K_r)

# =============================================================================
# 3. Calcul de la taille du filament 
# =============================================================================

Isb_resize_norm=255*((Isb_resize-Isb_resize.min())/(Isb_resize.max()-Isb_resize.min()))
profil=np.sum(Isb_resize_norm,axis=0)
counts,vals=np.histogram(profil,bins=int(np.max(profil)),range=(np.min(profil),np.max(profil)))

valeur_haute=vals[np.argmax(counts)]
valeur_basse=np.min(profil)
y_taille_fila=(valeur_haute-valeur_basse)/2 

#Interpolation des points pour si on trace droite et pas assez de points on trouve qqc
from scipy import interpolate

xint=(np.linspace(0,4096,4096*2))
xdata=np.linspace(0,4096,4096)

f = interpolate.interp1d(xdata, profil, fill_value='extrapolate')
profil_extra=f(xint)

plt.figure('Correction du profil des colonnes',(8,6))
plt.title('Courbe interpolée et corigée')
plt.plot(profil_extra,'xb')
plt.axhline(y_taille_fila,c='r',label='Mi-hauteur') # droite du y de la taille du filament 
plt.legend()
plt.show()

# Déterminer intersection entre y_taille_fila et profil_extra
taille_fila=np.argwhere(y_taille_fila>=profil_extra)  #Condition lorsque courbe au dessus de l'autre + indexation
x1=int(taille_fila[0])   #Valeur minimale 
x2=int(taille_fila[-1])   #Valeur maximale 

taille_filament=((x2-x1)*p)/K_h
print('La taille du filament est de',taille_filament, 'm')

