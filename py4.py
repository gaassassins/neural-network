from urllib.request import urlopen
import numpy as np

filename = input()
f = urlopen(filename)
sbux = np.loadtxt(f,skiprows=1, delimiter=",")
print(sbux.mean(axis=0))
