# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:35:44 2022

@author: p2105542
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import scipy.stats as st
import statsmodels.api as sm
import math as math


Air=pan.read_csv('http://tinyurl.com/y39an7ef/Data79899.csv',sep='\t',na_values='-')
#Air=pan.read_csv("data.csv",sep='\t',na_values='-')

###################################################Exercice 1. Statistiques#####################################################
#############################################################A.1################################################################
#Pour obtenir le type de chaque colonne dans un dataframe de pandas, On utilise la méthode dtypes
print("Le type des variables: ",Air.dtypes)
#############################################################A.2################################################################
#Maintenant on va créer un nouveau dataframe 'AirLogic' a partir de 'Air' à l'aide de la méthode 'isna()' Afin d'avoir des val-
#eurs booléens ('True' dans le cas ou y'a pas d'observation "nan" et 'False' Si il y'a d'observation)
AirLogic = Air.isna()

#On a créer ici un nouveau dataframe 'ISNA' (Meme que AirLogic), aprés on a supprimer tout les jours qui non pas d'observation
ISNA=pan.isna(Air)
for i in ['Feyzin ZI Acetylene','Feyzin ZI Benzène','Feyzin ZI Ethane','Feyzin ZI Ethylene','Feyzin ZI Isoprene','Pierre-Bénite Acetylene','Pierre-Bénite Benzène','Pierre-Bénite Ethane','Pierre-Bénite Ethylene','Pierre-Bénite Isoprene']:
    ISNA=ISNA[(ISNA[i] == False)]

#On affiche les tailles des deux dataframes 'Air' et 'ISNA'
print("Le nombre de jours d’observation de l’échantillon: ",len(Air))
print("Le nombre de jours où tous les composés organiques volatils ont été mesuré: ",len(ISNA))
#############################################################A.3################################################################

#Ici ona créer et insérer 2 nouvels colonnes dans le dataframe AirLogic et on l'ai a rempli avec 
#des 0 pour le moment

AirLogic.insert(13, "BenzèneOb", 0, True)
AirLogic.insert(14, "AutresOb", 0, True)

#Ici on a fait une boucle afin de vérifier les conditions 'Si le benzene était observé dans les
#deux stations' et si c'est vérifier, on affecte la valeur 'Vrai' dans la collones 'BenzeneOb'
#Pour le cas contraire, On affecte une valeur 'Faux'
N=len(AirLogic)
for i in range (0,N):
    #On utilise le dataframe AirLogic pour la verification (Observé = False / 'nan' = True)
    if (AirLogic['Feyzin ZI Benzène'][i] == False and AirLogic['Pierre-Bénite Benzène'][i] == False):
          AirLogic['BenzèneOb'][i] = "VRAI"
    else: AirLogic['BenzèneOb'][i] = "FAUX"
    
#On fait pareil que la premiere boucle mais pour les Autres polluants
for i in range (0,N):
    if (AirLogic['Feyzin ZI Acetylene'][i] == False and AirLogic['Pierre-Bénite Acetylene'][i] == False and AirLogic['Feyzin ZI Dioxyde soufre'][i] == False and AirLogic['Pierre-Bénite Dioxyde soufre'][i] == False and AirLogic['Feyzin ZI Ethylene'][i] == False and AirLogic['Pierre-Bénite Ethylene'][i] == False and AirLogic['Feyzin ZI Isoprene'][i] == False and AirLogic['Pierre-Bénite Isoprene'][i] == False):
          AirLogic['AutresOb'][i] = "VRAI"
    else: AirLogic['AutresOb'][i] = "FAUX"

#Ici on copie le dataframe 'AirLogic' dans un nouveau dataframe 'Observation'
Observation = AirLogic.copy()
    
#Ici on fait une boucle afin de supprimer toutes les colonnes apart les deux nouvelles
for i in ['Date','Feyzin ZI Acetylene','Feyzin ZI Benzène','Feyzin ZI Ethane','Feyzin ZI Ethylene','Feyzin ZI Isoprene','Pierre-Bénite Acetylene','Pierre-Bénite Benzène','Pierre-Bénite Ethane','Pierre-Bénite Ethylene','Pierre-Bénite Isoprene','Feyzin ZI Dioxyde soufre','Pierre-Bénite Dioxyde soufre']:
    del Observation[i]

#Maintenant on fait crée la table de contingence des deux nouvelles variables
data_crosstab = pan.crosstab(Observation['BenzèneOb'],
                            Observation['AutresOb'],
                               margins = False)
print(data_crosstab)
#############################################################A.4################################################################
#Ici on utiliser la méthode dropna pour se débarasser des jours de non observation pour les mésures de Feyzin ZI Benzène et
#Pierre-Bénite Benzène'
JoursObsB=Air.dropna(subset=['Feyzin ZI Benzène','Pierre-Bénite Benzène'])

#Ici on a créer une table de déscription qui donne différentes parameteres (Moyenne, quartile......) pour toutes les variables 
Resume=JoursObsB.describe()

#On fait calculer et afficher ligne par ligne toute les parametres demandés pour la variable 'Feyzin ZI Benzène'
print("La moyenne: ", np.mean(JoursObsB['Feyzin ZI Benzène']))#moyenne
print("La variance empirique: ", np.var(JoursObsB['Feyzin ZI Benzène']))#variance empirique
print("La variance empirique non-biaisé: ", np.var(JoursObsB['Feyzin ZI Benzène'],ddof=1))#variance empirique non biaisée
print("L'écart type': ", np.std(JoursObsB['Feyzin ZI Benzène'],ddof=1))#écart-type
print("La médiane", np.median(JoursObsB['Feyzin ZI Benzène']))#médiane
np.quantile(JoursObsB['Feyzin ZI Benzène'], [0.25,0.5,0.75],interpolation="lower")
#############################################################A.5################################################################
#On fait supprimer les lignes ou y'a pas d'observation dans les deux stations pour les mésures de Dioxyde soufre
Souffre=Air.dropna(subset=['Feyzin ZI Dioxyde soufre','Pierre-Bénite Dioxyde soufre'])

#Calculer la moyenne des valeurs des colonnes 'Feyzin ZI dioxyde de soufre' et 'Pierre-Bénite dioxyde de soufre' en ignorant les valeurs NaN
mean = (Souffre['Feyzin ZI Dioxyde soufre'] + Souffre['Pierre-Bénite Dioxyde soufre']) / 2

#on ajoute la variable 'SouffreMoyen' au dataframe
Souffre['SouffreMoyen'] = mean

#On affiche la moyenne de la variable 'SouffreMoyen'
print("La moyenne de la variable SouffreMoyen est : ", np.mean(Souffre["SouffreMoyen"]))
#############################################################A.6################################################################
#Une fonction qui calcule les parametres afin de déterminer les bornes de l'intervalle
#Cette fonction prend une variable X en parametre et un alpha (De niveau de confiance)
def Intervalle(x,alpha=0.05):
    m=np.mean(x)
    s=np.std(x,ddof=1)
    l=len(x)
    delta=(st.t.ppf(1-alpha/2,df=l-1)*s)/np.sqrt(l)
    return (m-delta,m+delta)

#Utilisation de la fonction pour la variable SouffreMoyen avec un alpha de 0.05 (Un niveau de confiance de 95%)
Intervalle(x=Souffre['SouffreMoyen'],alpha=0.05)
#############################################################A.7################################################################
#Début de la 1er boucle qui fait parcourir tout les variables
for ni in np.arange(0,12,1):
    nomD=['Feyzin ZI Acetylene','Feyzin ZI Benzène','Feyzin ZI Ethane','Feyzin ZI Ethylene','Feyzin ZI Isoprene','Pierre-Bénite Acetylene','Pierre-Bénite Benzène','Pierre-Bénite Ethane','Pierre-Bénite Ethylene','Pierre-Bénite Isoprene','Feyzin ZI Dioxyde soufre','Pierre-Bénite Dioxyde soufre']
    nomVar=nomD[ni]
    SemaineNan=False
    #fin de grupes de 7 a 731
    i=3
    #Début de la 2éme boucle qui donne l’indice des lundis de chaque semaine
    while i<731 and SemaineNan == False:
        nbNAN=0
        #i est lundice du lundi
        #Début de la 3éme boucle qui fait parcourir chaque jour de la semaine en comptant le nombre de ‘nan’
        for j in range (i,i+7):
            #on compte si la valeur est nan pour chaque jour de la semaine
            m=np.isnan(Air[nomVar][j])
            if(m==True):
                nbNAN=nbNAN+1
        if nbNAN == 7:
            SemaineNan=True
        i=i+7
    print("La variable",nomVar,"a au moins une semaine sans observation? ")
    print(SemaineNan)
    
#Début de la partie création du dataframe 'dfh'
dfh = Air[['Date','Feyzin ZI Acetylene','Feyzin ZI Benzène','Feyzin ZI Ethane','Feyzin ZI Ethylene','Feyzin ZI Isoprene']]

AirSept = Air.copy()

for i in ['Pierre-Bénite Acetylene','Pierre-Bénite Benzène','Pierre-Bénite Ethane','Pierre-Bénite Ethylene','Pierre-Bénite Isoprene','Feyzin ZI Dioxyde soufre','Pierre-Bénite Dioxyde soufre']:
    del AirSept[i]

#Début de la partie qui fait remplacé par es moyennes hebdomadaires de pollutions, pour chacune des 5 mesures
dfhSept = AirSept.copy()

dfhSept.set_index('Date', inplace=True)

dfhSept.index = pan.to_datetime(dfhSept.index)

dfh = dfhSept.resample('W').mean()
#Fin de la partie création du dataframe 'dfh'
#############################################################B.1################################################################

#On fait insérer les deux nouvelles colonnes qui contients les log des mésures de benzene et Ethylene
dfh.insert(5, 'logarithme de la mesure du Benzène', 0, True)
dfh.insert(6, 'logarithme de la mesure de l’Éthylène', 0, True)

#On calcule le logarithme de base exponentielle des mesures (Logarithme népérien)
N=len(dfh)
for i in range (0,N):
    dfh['logarithme de la mesure du Benzène'][i]=math.log(dfh['Feyzin ZI Benzène'][i],math.e)
    dfh['logarithme de la mesure de l’Éthylène'][i]=math.log(dfh['Feyzin ZI Ethylene'][i],math.e)

#On crée le model en question avec le y'mesure de l’Éthylène' et le x'mesure du Benzène'
model = sm.OLS(dfh['logarithme de la mesure de l’Éthylène'], dfh['logarithme de la mesure du Benzène']).fit();print(model.summary())

#On affecte les séries de valeurs aux x et y avec ce qui correspond
x = dfh['logarithme de la mesure du Benzène']
y = dfh['logarithme de la mesure de l’Éthylène']

#La partie du traçage du nuage de points avec la droite de regression avec les deux droites donnant les bornes de l’intervalle
#de prédiction au niveau 80%
prediction=model.get_prediction().summary_frame(alpha=0.2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "o", label="data")
ax.plot(x,  prediction["mean"], label="OLS",color="blue")#droite de régression
ax.plot(x, prediction["obs_ci_lower"], color="red")#borne inf de la prédiction
ax.plot(x, prediction["obs_ci_upper"], color="red")#borne sup  de la prédiction
ax.legend(loc="best")
fig.suptitle("Régression du logarithme de la mesure de l’Éthylène en fonction de logarithme de la mesure du Benzène avec intervalle de prédiction (rouge)")
#############################################################B.2################################################################
#Ici on prend l'image par exponentielle de la régression linéaire de x et de y
xx= dfh['logarithme de la mesure du Benzène'].apply(lambda x: math.exp(x))
yy= dfh['logarithme de la mesure de l’Éthylène'].apply(lambda x: math.exp(x))

#On crée un nouveau model a partir des nouveau xx et yy (qui sont des image exponentielle)
model2 = sm.OLS(yy,xx).fit();print(model.summary())

prediction2=model2.get_prediction().summary_frame(alpha=0.2)

#Cette partie concerne le tracage des nuages de points ainsi que la courbe de regression
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(xx, yy, "o", label="data")
ax.plot(xx,  prediction2["mean"], label="OLS",color="blue")#droite de régression
ax.plot(xx, prediction2["obs_ci_lower"], color="red")#borne inf de la prédiction
ax.plot(xx, prediction2["obs_ci_upper"], color="red")#borne sup  de la prédiction
ax.legend(loc="best")
fig.suptitle("Régression du l'image par exponentielle du logarithme de la mesure de l’Éthylène en fonction de l'image par exponentielle du logarithme de la mesure du Benzène avec intervalle de prédiction (rouge)")
#############################################################B.3################################################################
#On calcule les intervalles de prédiction a l'aide de la methode get_prediction avec fixation benzene au seuil réglémenté d=1
model.get_prediction(exog=[1]).summary_frame(alpha=0.2)
#############################################################Exercice 2.################################################################
#################################################################1.#####################################################################
plt.show()
N = 1000000  # taille du vecteur U
#on cree le verteur U
U = st.uniform.rvs(-10,5, size=N)
plt.hist(U)
#on crée le vecteur V
V = U[U < 0]
plt.hist(V)
xmin, xmax = -10, -5

plt.plot([xmin, xmax], [100000, 100000], 'k--')
plt.title('Histogramme associé à V')
#################################################################2.#####################################################################
N = 10000
U=st.uniform.rvs(0,1,size=N)
V=st.uniform.rvs(0,1,size=N)
U0 = U
V0 = np.exp(-U0**2/2)*V
U1=-(10*np.log(U))
V1=U*V

#on trace le nuage de points
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(U0, V0, "o", label="points")
fig.suptitle("Nuage de pointsde l’échantillon (U0, V0) points")




#on trace le nuage de points
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(U0, V0, "o", label="points")
fig.suptitle("Nuage de pointsde l’échantillon (U0, V0) points .")

# Définir les demi-axes de l'ovale
a = 1.254  
b = 1.01
#on genere les points pour tester l'ecuation
U=st.uniform.rvs(size=10000,loc=-1,scale=2) 
V=st.uniform.rvs(size=10000,loc=-1,scale=2)

#c'est l'équation lour les poins dans l'aire d'une ellipse (x^2)/a^2 + (y^2)/b^2 <= 1
S=U**2/a**2+V**2/b**2  #ellipse 
Cas =(S<1)  # condition  
U0=U[Cas]    # elts qui verifient rayon <1
V0 = V[Cas]   # elts qui verifient rayon <1
plt.axis('equal')
plt.plot(U,V,marker=',',linestyle='')

#l'échantillon est uniformement distribué sur [-1,1][-1,1]
plt.axis('equal')
plt.plot(U0,V0,marker=',',linestyle='')

#l'echantillon du couple (U0,V0) est une ellipse de centre 0,0 de rayon 1 qui a 0<U0<1 et 0<V0<1

################
N = 10000
U=st.uniform.rvs(0,1,size=N)
V=st.uniform.rvs(0,1,size=N)
U0 = U
V0 = np.exp(-U0**2/2)*V
U1=-(10*np.log(U))
V1=U*V
#on trace le nuage de points
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(U1, V1, "o", label="points")
fig.suptitle("Nuage de pointsde l’échantillon (U1, V1)")

#################################################################3.#####################################################################
N=50000
Y=st.uniform.rvs(0,1,8*N);Y.shape=(8,N)
T=np.repeat(0.1,2*N);T.shape=(2,N)
X=(3*Y[6,]<1)
Z=(3*Y[6,]>2)
T[0,]=Y[0,]*X+(Y[2,]*3/2+1)*(1-X)*(1-Z)+(2.5-np.exp(25/8)*np.log(Y[4,]))*Z
T[1,]=(Y[1,]*X+(Y[3]*2/3)*(1-X)*(1-Z)+(np.exp(-25/8)*Y[4,]*Y[5,])*Z)
Cas=(T[1,]<np.exp(-T[0,]**2/2))
R1 = T[0,Cas]
S = T[1,Cas]
R2 = R1*(2*(2*Y[7,Cas]>1)-1)

# Tracer le nuage de points
plt.show()
plt.scatter(T[0,], T[1,])
plt.show()

# Filtrer l'échantillon en ne conservant que les vecteurs vérifiant T[0,] < 2.5
T_below_2_5 = T[:, np.where(T[0,] < 2.5)[0]]

# Filtrer l'échantillon en ne conservant que les vecteurs vérifiant T[0,] > 2.5
T_above_2_5 = T[:, np.where(T[0,] > 2.5)[0]]

# Tracer le nuage de points pour T_below_2_5
plt.scatter(T_below_2_5[0,], T_below_2_5[1,])
plt.show()

# Tracer le nuage de points pour T_above_2_5
plt.scatter(T_above_2_5[0,], T_above_2_5[1,])
plt.show()
#################################################################4.#####################################################################
# Tracer le nuage de points pour T
plt.scatter(T[0,], T[1,])

# Tracer le nuage de points pour R1 et S
plt.scatter(R1, S, color='red')

# Restreindre l'axe des abscisses à [0, 5]
plt.xlim(0, 5)

plt.show()

import seaborn as sns
#On install d'abord avec %pip install seaborn

# Calculer la moyenne et l'écart-type de R2
mean_R2 = np.mean(R2)
std_R2 = np.std(R2)

# Tracer une densité de probabilité de R2
sns.kdeplot(R2, shade=True)

# Afficher la moyenne et l'écart-type de R2
plt.axvline(mean_R2, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_R2 + std_R2, color='red', linestyle='dotted', linewidth=1)
plt.axvline(mean_R2 - std_R2, color='red', linestyle='dotted', linewidth=1)

plt.show()
#################################################################5.#####################################################################
# Tracer le nuage de points pour R2 et S
plt.scatter(R2, S)

plt.show()

# Tracer une densité de probabilité de R2
sns.kdeplot(R2, shade=True)

plt.show()

# Tracer une densité de probabilité de R2
sns.kdeplot(R2, shade=True)

# Tracer une courbe du bord entre -2 et 2
plt.fill_between(x=np.arange(-2, 2, 0.01), y1=0, y2=1, color='red', alpha=0.1)

plt.show()
#################################################################6.#####################################################################
# Tracer un histogramme de R2
plt.hist(R2, bins=50)

# Tracer une densité de probabilité d'une loi normale
sns.kdeplot(R2, shade=True)

# Créer une figure à deux tracés
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Tracer un histogramme de R2 dans le premier tracé
ax[0].hist(R2, bins=50)

# Tracer une densité de probabilité d'une loi normale dans le second tracé
sns.kdeplot(R2, shade=True, ax=ax[1])

plt.show()
#################################################################Fin#####################################################################

