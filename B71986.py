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
