import numpy as numpy
import matplotlib.pyplot as plt

greyhounds = 500
laborado = 500

grey_height = 28 + 4 * numpy.random.randn(greyhounds) 
lab_height = 24 + 4 * numpy.random.randn(laborado)

plt.hist([grey_height,lab_height], stacked = True, color = ['g','b'])
plt.show()