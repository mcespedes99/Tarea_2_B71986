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
a = df.to_numpy()
plt.hist(a, bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
plt.show()
