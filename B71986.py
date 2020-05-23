# -*- coding: utf-8 -*-
"""
Tarea #2. IE0405 - Modelos Probabilísticos de Señales y Sistemas.
Empezada el Viernes 22 de Mayo 14:34 2020

@author: Mauricio Céspedes Tenorio.
Carné: B71986
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('datos.csv',header= None)
#pmf = df[0].value_counts(normalize=True)
#plt.bar(list(pmf.index), pmf.values)
#pmf.plot.hist(bins=100)
#df = pd.read_csv('datos.csv',header= None)
a = df.to_numpy()
plt.hist(a,20,(0,100),density=True) 
plt.show()

plt.show()
