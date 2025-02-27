#--------------------------------------------------------
# Heliocentric coordinates conversion functions
# (x, v <-> elliptical elements)
# (c) ASD/IMCCE
# 21/01/2015 : J-B Delisle fortran to python convertion
#--------------------------------------------------------

#--------------------------------------------------------
# Notations and units:
# x[0:2] = positions [AU]
# xd[0:2] = velocities [AU/yr]
# cmu = G*(m_star + m_planet) [AU^3/yr^2]
# ell[0:5] :
# - a [AU]
# - lambda (mean longitude) [rad]
# - k = e * cos(w) (w: longitude of perihelion)
# - h = e * sin(w)
# - q = sin(i/2) cos(Omega) (i: inclination, Omega: longitude of node)
# - p = sin(i/2) sin(Omega)
#--------------------------------------------------------

#--------------------------------------------------------
# Convertion functions:
# ell = cart2ell(x,xd,cmu)
# x,xd = ell2cart(ell,cmu)
#--------------------------------------------------------

import numpy as np

def cart2ell(x,xd,cmu):
  #--------------------------------------------------------
  #     Conversion position-vitesse en elements elliptiques
  #     convient pour les mouvements elliptiques non
  #     rectilignes ( C <> 0) et les inclinaisons differentes
  #     de Pi ( cos(i) <> -1)
  #     J. Laskar (1986-2004)  (voir notes de cours)
  #     version revisee par rapport a la version originale de 1986
  #     amelioration dans toutes les precisions
  #     par rapport a la version XYZKHQP1 modifiee par Fred.
  #     BUG corrige en longitude quand e est nul
  #     precision moins bonne (mais encore OK)
  #     pour tres grandes inclinaisons ( i > 1 radian)
  #     v. 1.00  (jxl 11/06/2004)
  #     ell : a, lambda, k, h, q, p
  #---------------------------------------------------------
  ell = np.empty(6)
  v = np.empty(3)
  #------------ normalisation des vitesses
  smu=np.sqrt(cmu)
  v[0]=xd[0]/smu
  v[1]=xd[1]/smu
  v[2]=xd[2]/smu
  #------------ quantitees utilies
  r=np.sqrt(x[0]**2+x[1]**2+x[2]**2)
  v2=v[0]**2+v[1]**2+v[2]**2
  rv=(x[0]*v[0]+x[1]*v[1]+x[2]*v[2])
  c1=x[1]*v[2]-x[2]*v[1]
  c2=x[2]*v[0]-x[0]*v[2]
  c3=x[0]*v[1]-x[1]*v[0]
  cc=c1**2+c2**2+c3**2
  dc= np.sqrt(cc)
  #------------ demi grand axe a
  aa = r/(2.0-r*v2)
  if aa<=0:
    return(np.array([float('nan') for _ in range(6)]))
  ell[0]=aa
  usqa= np.sqrt(2.0/r-v2)
  #------------ q,p
  aux0 = np.sqrt(2*(cc+dc*c3))
  ell[4]=-c2/aux0
  ell[5]= c1/aux0
  #------------ k,h
  #------------ coefs de  e * m^-1
  a11= v2-1.0/r
  a12= rv/(r*dc)
  a21=-rv
  a22= dc-r/dc
  #------------ calcul de (h,k)
  c11=x[0]*a11+v[0]*a21
  c12=x[0]*a12+v[0]*a22
  c21=x[1]*a11+v[1]*a21
  c22=x[1]*a12+v[1]*a22
  fac1=c1/(dc+c3)
  fac2=c2/(dc+c3)
  k1= c11-fac1*(x[2]*a11+v[2]*a21)
  h1=-c12+fac1*(x[2]*a12+v[2]*a22)
  h2= c21-fac2*(x[2]*a11+v[2]*a21)
  k2= c22-fac2*(x[2]*a12+v[2]*a22)
  h =(h1+h2)/2.0
  k =(k1+k2)/2.0
  ell[2]=k
  ell[3]=h
  #------------ calcul de la longitude
  b12=v[0]-fac1*v[2]
  b22=v[1]-fac2*v[2]
  aux1 = (r*v2-1.0)/(1.0+dc*usqa)
  sinf=-b12*r*usqa +h*aux1
  cosf= b22*r*usqa +k*aux1
  f=arctan2(sinf,cosf)
  ell[1]=f-rv*usqa
  return(ell)

def ell2cart(ell,cmu):
  #--------------------------------------------------------
  #     conversion elements elliptiques vers position-vitesse
  #     j. laskar 1986
  #     version sans les derivees partielles
  #     legerment remaniee le 11/6/2004
  #     sans aucun changement dans les resultats
  #--------------------------------------------------------
  x = np.empty(3)
  xp = np.empty(3)
  tx1 = np.empty(2)
  tx1t = np.empty(2)
  rot = np.empty((3,2))

  a=ell[0]
  l=ell[1]
  k=ell[2]
  h=ell[3]
  q=ell[4]
  p=ell[5]
  na=np.sqrt(cmu/a)
  phi=np.sqrt(1.0-k**2-h**2)
  ki =np.sqrt(1.0-q**2-p**2)
  #---- matrice de rotation ----------------------------------------------
  rot[0,0]=1.0-2.0*p**2
  rot[0,1]=2.0*p*q
  rot[1,0]=2.0*p*q
  rot[1,1]=1.0-2.0*q**2
  rot[2,0]=-2.0*p*ki
  rot[2,1]= 2.0*q*ki
  #---- calcul de la longitude excentrique f -----------------------------
  #---- f = anomalie excentrique e + longitude du periapse omegapi -------
  f=keplkh2(l,k,h)
  sf    =np.sin(f)
  cf    =np.cos(f)
  umrsa = k*cf+h*sf
  psilmf   = (-k*sf+h*cf)/(1.0+phi)
  psiumrsa =        umrsa/(1.0+phi)
  na2sr    = na/(1.0-umrsa)
  #---- calcul de tx1 et tx1t --------------------------------------------
  tx1[0] =a*(cf- psilmf*h-k)
  tx1[1] =a*(sf+ psilmf*k-h)
  tx1t[0]=na2sr*(-sf+psiumrsa*h)
  tx1t[1]=na2sr*( cf-psiumrsa*k)
  #---- calcul de xyz ----------------------------------------------------
  x[0]  =rot[0,0]*tx1[0]  + rot[0,1]*tx1[1]
  x[1]  =rot[1,0]*tx1[0]  + rot[1,1]*tx1[1]
  x[2]  =rot[2,0]*tx1[0]  + rot[2,1]*tx1[1]
  xp[0] =rot[0,0]*tx1t[0] + rot[0,1]*tx1t[1]
  xp[1] =rot[1,0]*tx1t[0] + rot[1,1]*tx1t[1]
  xp[2] =rot[2,0]*tx1t[0] + rot[2,1]*tx1t[1]
  return(x,xp)

def keplkh2(l,k,h):
  #---------------------------------------------------
  #     resolution de l'equation de kepler
  #     pour les variables l=m+pi et a=e+pi
  #     precision 2*(eps-mach) pour des excentricites
  #     inferieures a 0.3
  #     temps de calculs :
  #     pour la premiere iteration (recherche du depart)
  #                     2 (cos sin)
  #                    11 (* /)
  #                     9 (+ -)
  #     pour la deuxieme iteration
  #                     2 (cos sin)
  #                    25 (* /)
  #                    14 (+ -)
  #     f. joutel 26/4/99
  #     26/1/01 appel eventuel a la methode de newton
  #---------------------------------------------------
  eps = 2*2.26e-16
  imax = 20
  # depart methode d'ordre 3
  a=l
  ca=np.cos(a)
  sa=np.sin(a)
  se=k*sa-h*ca
  ce=k*ca+h*sa
  fa=a-se-l
  f1a=1.0-ce
  f2a=se/2.0
  f3a=ce/6.0
  d1=-fa/f1a
  d2=-fa/(f1a-d1*f2a)
  d3 =-fa/(f1a+d2*(f2a+d2*f3a))
  a=a+d3
  # methode d'ordre 6
  ca=np.cos(a)
  sa=np.sin(a)
  se=k*sa-h*ca
  ce=k*ca+h*sa
  fa=a-se-l
  f1a=1.0-ce
  f2a=se/2.0
  f3a=ce/6.0
  f4a=-se/24.0
  f5a=-ce/120.0
  d1=-fa/f1a
  d2=-fa/(f1a-d1*f2a)
  d3=-fa/(f1a+d2*(f2a+d2*f3a))
  d4=-fa/(f1a+d3*(f2a+d3*(f3a+d3*f4a)))
  d5=-fa/( f1a+d4*(f2a+d4*(f3a+d4*(f4a+d4*f5a))))
  a=a+d5
  #     return
  #     on teste la precision obtenue
  i=0
  while True:
    i=i+1
    ca=np.cos(a)
    sa=np.sin(a)
    se=k*sa-h*ca
    fa=a-se-l
    ce=k*ca+h*sa
    f1a=1.0-ce
    d1=-fa/f1a
    #     si la precison n'est pas bonne, on continue les calculs
    #     en iterant la methode d'ordre 1
    if (np.abs(d1)/np.max([1.0,np.abs(a)]) > eps):
      if (i > imax):
        #     write(*,*) 'erreur fatale dans elliptid:keplkh2'
        #     write(*,*) 'erreur :',abs(d1)/dmax1(1.0,abs(a))
        return(a)
      a=a+d1
    else:
      return(a)
