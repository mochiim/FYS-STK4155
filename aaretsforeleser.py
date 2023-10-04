import numpy as np
import matplotlib.pyplot as plt

nominasjoner = ['Arne Hole',
'Joakim Bergli',
'Anders Kvellestad',
'Morten Hjorth-Jensen',
'Tom Louis Lindstrøm',
'Frode Kristian Hansen',
'Frode Hansen',
'Clas Persson',
'Tom Lindstrøm',
'Are Raklev',
'Dorthea Gjestvang',
'Frode Kristian Hansen',
'Ann-Cecilie Larsen',
'Anders Kvellestad',
'Frode Kristian Hansen',
'Ann-Cecilie Larsen',
'Nicholas Ssessanga',
'Arne Hole',
'Tom Lindstrøm',
'Frode Hansen',
'Tom Louis Lindstrøm',
'Tom Lindstrøm',
'Fred-Johan Pettersen',
'Olav Fredrik Syljuåsen',
'Tom Lindstrøm',
'Tom Lindstrøm',
'Erik Adli',
'Frode Kristian Hansen',
'Tom Lindstrøm',
'Frode Strisland',
'Johannes Skaar',
'Anders Kvellestad',
'Dag Kristian Dysthe']

items_to_remove = ['Arne Hole', 'Tom Lindstrøm', 'Tom Louis Lindstrøm', 'Frode Hansen', 'Frode Kristian Hansen']
nominasjoner = list(set(nominasjoner) - set(items_to_remove))



plt.figure(figsize = (8, 10))
plt.hist(nominasjoner)
plt.xticks(rotation = 'vertical')
plt.show()