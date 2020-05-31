# -*- coding: utf-8 -*-
"""
Tarea #2. IE0405 - Modelos Probabilísticos de Señales y Sistemas.
Empezada el Viernes 22 de Mayo 14:34 2020

@author: Mauricio Céspedes Tenorio.
Carné: B71986
"""
#Librerías
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace, sqrt, arange, sqrt, exp, pi, max
from scipy.stats import rayleigh, norm

print("Respuestas de la Tarea #1 del curso IE0405 - Modelos Probabilísticos de Señales y Sistemas.")
print("Estudiante: Mauricio Céspedes Tenorio. Carné: B71986.")
#Extracción de datos del archivo datos.cvs en un dataframe (se especifica que no hay encabezado en el archivo):
df = pd.read_csv('datos.csv',header= None)
#Para poder graficar, se realizó la coversión del df a un array de numpy:
a = df.to_numpy()

"""Punto 4. Ploteo del histograma"""
print("\n\nPunto 4. Para este caso, se graficó el histograma.")
#Ploteo del histograma:
plt.hist(a,15,(0,a.max()),density=True)

"""Punto 5. Curva de mejor ajuste y su gráfica"""
#Se encontró visualmente que el histograma tenía una forma similar a la PDF de Rayleigh.
#Se encuentra el mejor fit (curva de mejor ajuste) de una función de distribución Rayleigh (esta función da como return la localización y escala de la curva de mejor ajuste):
print("\n\nPunto 5. La curva de mejor ajuste se hizo observando que el histograma presenta un forma similar a una PDF Rayleigh. Se ploteó dicha curva sobre el histograma en una misma Figura.")
parametros = rayleigh.fit(a)
#Definición de un espacio lineal entre 0 y el valor máximo de "a" con mil puntos.
x = linspace(0,a.max(),1000)
#Se crea una PDF Rayleigh con los datos obtenidos de la curva de mejor ajuste:
pdf_fit = rayleigh.pdf(x,loc=parametros[0],scale=parametros[1])
#Ploteo de la curva de mejor ajuste:
plt.plot(x,pdf_fit,'r-', label = "Modelo encontrado con ayuda de Scipy")
#Comando para abrir las gráficas (se añadieron títulos a los ejes y gráfico):
plt.xlabel('x')
plt.ylabel('fx(x)')
plt.title('Histograma y curva de mejor ajuste de datos dados')
plt.legend()
plt.show()

"""Punto 6. Encontrar la probabilidad en el intervalo [19, 86] en el modelo encontrado y contrastarlo con la frecuencia relativa de los elementos de datos.csv que están en realidad en ese mismo intervalo."""
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
print("\n\nPunto 6. Se calculó la probabilidad del intervalo [19,86] con la CDF del modelo encontrado y mediante frecuencia relativa.")
print("La probabilidad del intervalo [19,86] con el modelo obtenido es de: "+"{:.4f}".format(cdf_19y86*100)+"%.\nLa probabilidad obtenida con frecuencia relativa es de: "+"{:.4f}".format(frec_rel_19y86*100)+"%.\nEsto representa un error del ""{:.4f}".format(error*100)+"%, el cual es muy bajo.")

"""Punto 7. Cálculo de los primeros cuatro momentos:"""
media, var, skew, kurt = rayleigh.stats(loc=parametros[0],scale=parametros[1],moments='mvsk')
print("\n\nPunto 7. Se calcularon los primeros cuatro momentos con ayuda de stats de Scipy.")
print("i) La media de la PDF modelada es de: "+"{:.4f}".format(media)+". Lo cual indica que la mayoría de datos se encuentran alrededor de este punto. Esto coincide con lo visto en las gráficas.")
print("ii) La varianza de la PDF modelada es de: "+"{:.4f}".format(var)+". Por ende, la desviación estándar es: " +"{:.4f}".format(sqrt(var))+  ", que es del mismo orden de magnitud que la media; lo cual indica que los datos están muy dispersos, que es usual en una PDF Rayleigh.")
print("iii) La inclinación de la PDF modelada es de: "+"{:.4f}".format(skew)+". Esto implica que la curva está sesgada hacia la derecha (tomando como referencia su media). En la curva se observa que hay más datos a la derecha de 37,57. Además, coincide con el valor esperado de inclinación para una PDF Rayleigh (0,63 aproximadamente).")
print("iv) La kurtosis de la PDF modelada es de: "+"{:.4f}".format(kurt)+". Como es mayor que cero, implica que la curva tiene una cima prominente, lo cual es bastante evidente para PDF Rayleigh. Además, coincide con el valor esperador de Kurtosis para esta distribución (0,24 aproximadamente).")

"""Punto 8. Si los valores de datos.csv son X y pasa por la transformación Y = sqrt(X), graficar el histograma de Y. Encontrar la función de densidad de Y."""
print("\n\nPunto 8. Se grafica el histograma de Y=sqrt(X), la curva de mejor ajuste de la PDF y se imprime la forma de dicha PDF de mejor ajuste.")
#Primero, se encuentra la matriz con los valores de Y, que son la raíz de los valores de la matriz a:
y = sqrt(a)
#Ploteo del histograma de y:
plt.hist(y,15,(0,y.max()),density=True)
#Parámetro de curva de mejor ajuste para Y (del histograma se observó que parece una de distribución normal)
param_norm = norm.fit(y)
#Definición de un espacio lineal entre 0 y el valor máximo de "y" con 100 puntos (se observó del histograma, que la curva está en este intervalo).
x_2 = linspace(0,y.max(),100)
#Se crea una PDF normal con los datos obtenidos de la curva de mejor ajuste:
pdf_norm_fit = norm.pdf(x_2,loc=param_norm[0],scale=param_norm[1])
#Ploteo de la curva de mejor ajuste (puntos extra):
plt.plot(x_2,pdf_norm_fit,'r-', label = "Modelo encontrado con ayuda de Scipy")
#Encontré la función de dicha PDF normal y la grafiqué con puntos azules en la misma figura para corroborar que estuviera bien:
x_3 = arange(0,y.max(),0.3)
plt.plot(x_3,exp(-((x_3-param_norm[0]))**2/(2*param_norm[1]**2))/(sqrt(2*pi*param_norm[1]**2)),'bo', label='Función encontrada "a mano"')
#Graficación de PDF normal encontrada con la fórmula respectiva (teoría):
plt.plot(x_3,(2*x_3*(x_3**2-parametros[0]) / parametros[1]**2)* exp(-(x_3**2 -parametros[0])**2/(2*(parametros[1]**2))),'k--', label='Función encontrada con fórmula')
#Impresión de la forma de esta función en terminal:
print("La expresión para función de densidad de Y (PDF normal) es de la forma: f_y(y)=e**[(y-5.90638504488997)^2 /(2*2.646358282)] / (sqrt(2*pi*2.646358282)). Donde 5.9064 es la media y 2.64636 es la varianza.")
#Comando para abrir las gráficas (se añadieron títulos a los ejes y gráfico y leyendas):
plt.xlabel('y')
plt.ylabel('fy(y)')
plt.title('Histograma y curva de mejor ajuste de "y"')
plt.legend()
plt.show()
