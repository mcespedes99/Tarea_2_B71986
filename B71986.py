# -*- coding: utf-8 -*-
"""
Tarea #2. IE0405 - Modelos Probabilísticos de Señales y Sistemas.
Empezada el Viernes 22 de Mayo 14:34 2020

@author: Mauricio Céspedes Tenorio.
Carné: B71986
"""
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from scipy.stats import rayleigh

#Extracción de datos del archivo datos.cvs en un dataframe (se especifica que no hay encabezado en el archivo):
df = pd.read_csv('datos.csv',header= None)
#Para poder graficar, se realizó la coversión del df a un array de numpy:
a = df.to_numpy()

"""Punto 4. Ploteo del histograma"""
plt.hist(a,15,(0,100),density=True)

"""Punto 5. Curva de mejor ajuste y su gráfica"""
#Se encontró visualmente que el histograma tenía una forma similar a la PDF de Rayleigh.
#Se encuentra el mejor fit (curva de mejor ajuste) de una función de distribución Rayleigh (esta función da como return la localización y escala de la curva de mejor ajuste):
parametros = rayleigh.fit(a)
#Definición de un espacio lineal entre 0 y 100 con mil puntos.
x = linspace(0,100,1000)
#Se crea una PDF Rayleigh con los datos obtenidos de la curva de mejor ajuste:
pdf_fit = rayleigh.pdf(x,loc=parametros[0],scale=parametros[1])
#Ploteo de la curva de mejor ajuste:
plt.plot(x,pdf_fit,'r-')
#Comando para abrir las gráficas:
plt.show()

"""Encontrar la probabilidad en el intervalo [19, 86] en el modelo encontrado y contrastarlo con la frecuencia relativa de los elementos de datos.csv que están en realidad en ese mismo intervalo."""
#Probabilidad con la curva encontrada: resta de la cdf en 86 y la cdf en 19 con los datos de mejor ajuste
cdf_19y86 = rayleigh.cdf(86,loc=parametros[0],scale=parametros[1])-rayleigh.cdf(19,loc=parametros[0],scale=parametros[1])
#Probabilidad con frecuencia relativa:
#i) Primero se hace un DataFrame con los datos en el intevalo [19, 86]:
df_entre_19y86 = df[(df[0]<=86) & (df[0]>=19)]
#ii) Cantidad de datos totales del csv:
N_total = len(df)
#iii) Cantidad de datos totales del csv entre 19 y 86:
N_entre_19y86 = len(df_entre_19y86)
#iv) Frecuencia relativa (probabilidad en el intervalo [19,86]):
frec_rel_19y86 = N_entre_19y86/N_total
#Error de la probabilidad obtenida con el modelo:
error = abs(cdf_19y86-frec_rel_19y86)/frec_rel_19y86
print("La probabilidad en el intervalo [19,86] con el modelo dato es de: "+"{:.4f}".format(cdf_19y86*100)+"%.\nLa probabilidad obtenida con frecuencia relativa es de: "+"{:.4f}".format(frec_rel_19y86*100)+"%.\nEsto representa un error del ""{:.4f}".format(error*100)+"%, el cual es muy bajo.")
