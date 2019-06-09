
#https://lectures.quantecon.org/py/about_py.html


############################
#Python Essentials
############################

x = True
x

y = 100 < 10
y

type(y)

x + y

x * y

True + True

bools = [True, True, False, True]  # List of Boolean values

sum(bools)

a, b = 1, 2
c, d = 2.5, 10.0
type(a)
type(c)

#complex numbers
x = complex(1, 2)
y = complex(2, 1)
x * y


#Containers
#tuples
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
type(x)

#Python lists are mutable
x = [1, 2]
x[0] = 10
x

x = (1, 2)
x[0] = 10

#Tuples (and lists) can be “unpacked” as follows
integers = (10, 20, 30)
x, y, z = integers
x

list1 = [10,20,30]
type(list1)
x, y, z = list1

#Slice Notation
a = [2, 4, 6, 8]
a[1:] #1 onwards
a[2:]
a[1:3]
#The general rule is that a[m:n] returns n - m elements, starting at a[m]

a[-2:]  # Last two elements of the list

s = 'foobar'
s[-3:]  # Select the last three elements

integers = (10, 20, 30)
integers[0:2]
integers[-1:]

#Sets and Dictionaries
d = {'name': 'Frodo', 'age': 33}
type(d)

d['age']
d['name']

print(d['age'])
print(d['name'])

d = {'name': 'Frodo', 'age': 33, 'friend': 'Sam Wise',
    'Surname': 'Bagins'}
print(d['friend'])
print(d['Surname'])


#Sets are unordered collections without duplicates,
#and set methods provide the usual set theoretic operations
s1 = {'a', 'b'}
type(s1)

s2 = {'b', 'c'}
s1.issubset(s2)

s3 = {'b'}
s3.issubset(s1)

s1.intersection(s2)


s3 = set(('foo', 'bar', 'foo'))
s3

aux1 = ('foo', 'bar', 'foo')
type(aux1)

s4 = set(aux1)
type(s4)
s4


#Input and Output

f = open('newfile.txt', 'w')   # Open 'newfile.txt' for writing
f.write('Testing\n')           # Here '\n' means new line
f.write('Testing again')
f.close()

%pwd

f = open('newfile.txt', 'r') #'r' means that we want to read the file
out = f.read()
out
print(out)

f = open('C:\\Users\\willi\\Desktop\\working\\projects_git\\python-with-QuantEcon\\newfile.txt', 'r')
out = f.read()
print(out)

#Iteraing

%%file us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229

data_file = open('us_cities.txt', 'r')
for line in data_file:
    city, population = line.split(':')         # Tuple unpacking
    city = city.title()                        # Capitalize city names
    population = f'{int(population):,}'        # Add commas to numbers
    print(city.ljust(15) + population)
data_file.close()

x = ('a', 'b') #tuple, immutable list










############################
#An Introductory Example
############################


#Exercise 2

from numpy.random import uniform
def binomial_rv(n, p):
    aux_list = []
    for i in range(n):
        aux_e = uniform()
        aux_e2 = int(aux_e > p)
        aux_list.append(aux_e2)
    return aux_list

binomial_rv(10,.5)


def binomial_rv2(n, p):
    aux_list = []
    i = n
    while i > 0 :
        aux_e = uniform()
        aux_e2 = int(aux_e > p)
        aux_list.append(aux_e2)
        i = i - 1
    return aux_list

binomial_rv2(10,.5)



def binomial_rv3(n, p):
    aux_1 = [uniform() for i in range(n)]
    aux_list = [ int(x > p)  for x in aux_1]
    return aux_list

binomial_rv3(10,.5)


#Exercise 3

#failed attemp
import numpy as np
def pi(n):
    aux_1 = [uniform() for i in range(n)]
    aux_2 = [(1-x**2)**.5 for x in aux_1]
    aux_3 = [ for x in aux_2]

#failed attemp
def pi(n):
    aux_1 = [x/n for x in range(n)]
    aux_2 = [(1-x**2)**.5 for x in aux_1]
    aux_2_1 = [uniform() for i in range(n)]
    aux_3 = [ for x in aux_2]

#solution
n = 100000
count = 0
for i in range(n):
    u, v = np.random.uniform(), np.random.uniform()
    d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
    if d < 0.5:
        count += 1
area_estimate = count / n
print(area_estimate * 4)  # dividing by radius**2


n = 100000
count = 0
for i in range(n):
    u, v = np.random.uniform(), np.random.uniform()
    d = np.sqrt(u**2+v**2)
    if d < 1:
        count += 1
area_estimate = count / n
print(area_estimate*4)

#in the future try to verify which algorithm is better...


#Exercise 4
from numpy.random import uniform
aux_1 = [uniform() for i in range(10)]
aux_2 = [int(i<.5) for i in aux_1]
val_1 = 0
val_2 = 0
for i in aux_2:
    if i==1:
        val_1 = 1 + val_1
    if val_1 == 3:
        val_2=1
    if i==0 :
        val_1 = 0
print(val_2)
print(aux_2)

#solution from the book
from numpy.random import uniform
payoff = 0
count = 0
for i in range(10):
    U = uniform()
    count = count + 1 if U < 0.5 else 0
    if count == 3:
        payoff = 1
print(payoff)


#Exercise 5
import numpy as np
import matplotlib.pyplot as plt
T = 200
alpha=0.9
series = []
aux_1 = [np.random.randn() for i in range(T)]
val_1=0
for i in aux_1:
    val_1 = i + alpha*val_1
    series.append(val_1)
plt.plot(series)
plt.show()


#solution from the book
α = 0.9
ts_length = 200
current_x = 0
x_values = []
for i in range(ts_length + 1):
    x_values.append(current_x)
    current_x = α * current_x + np.random.randn()
plt.plot(x_values)
plt.show()


#Exercise 6


import numpy as np
import matplotlib.pyplot as plt

x = [np.random.randn() for i in range(100)]
plt.plot(x, label="white noise")
plt.legend()
plt.show()


alphas = [0,0.8,0.98]
ts_length = 200
for alpha in alphas:
    current_x = 0
    x_values = []
    for i in range(ts_length + 1):
        x_values.append(current_x)
        current_x = alpha * current_x + np.random.randn()
    plt.plot(x_values, label="α ="+str(alpha))
plt.legend()
plt.show()


#solution from the book
αs = [0.0, 0.8, 0.98]
ts_length = 200

for α in αs:
    x_values = []
    current_x = 0
    for i in range(ts_length):
        x_values.append(current_x)
        current_x = α * current_x + np.random.randn()
    plt.plot(x_values, label=f'α = {α}')
plt.legend()
plt.show()




αs = [0.0, 0.8, 0.98]
ts_length = 200

for α in αs:
    x_values = []
    current_x = 0
    for i in range(ts_length):
        x_values.append(current_x)
        current_x = α * current_x + np.random.randn()
    plt.plot(x_values, label='α ='+str(α) )
plt.legend()
plt.show()



#########################################
#Setting up Your Python Environment
#########################################

#Version 1
import numpy as np
import matplotlib.pyplot as plt
%matplotlib tk

x = np.random.randn(100)
plt.plot(x)
plt.show()


#this text show some examples of what python can do

import numpy as np                     # Load the library

a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
b = np.cos(a)                          # Apply cosine to each element of a
c = np.sin(a)                          # Apply sin to each element of a

b @ c                                  #take the inner product

from scipy.stats import norm
from scipy.integrate import quad

ϕ = norm()
value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
value

###############################################################################
#Symbolic Algebra##############################################################
###############################################################################
#It’s useful to be able to manipulate symbolic expressions, as in Mathematica or Maple
from sympy import Symbol

x, y = Symbol('x'), Symbol('y')  # Treat 'x' and 'y' as algebraic symbols
x + x + x + y

expression = (x + y)**2
expression.expand()

expr1 = (x + y)**5
expr1.expand()

z = Symbol('z')
expr2 = (x+y+z)**2
expr2.expand()

#solve polynomials
from sympy import solve
solve(x**2 + x + 2)

#calculate limits, derivatives and integrals
from sympy import limit, sin, diff
limit(1 / x, x, 0)
limit(1 / x, x, 1)

limit(sin(x) / x, x, 0)

diff(sin(x), x)

###############################################################################
#Statistics ###################################################################
###############################################################################
import pandas as pd
np.random.seed(1234)

data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
dates = pd.date_range('28/12/2010', periods=5)

df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
print(df)

df.mean()

###############################################################################
#Networks and Graphs ##########################################################
###############################################################################
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#%matplotlib inline
%matplotlib tk
np.random.seed(1234)

# Generate random graph
p = dict((i,(np.random.uniform(0, 1),np.random.uniform(0, 1))) for i in range(200))
G = nx.random_geometric_graph(200, 0.12, pos=p)
pos = nx.get_node_attributes(G, 'pos')

# find node nearest the center point (0.5, 0.5)
dists = [(x - 0.5)**2 + (y - 0.5)**2 for x, y in list(pos.values())]
ncenter = np.argmin(dists)

# Plot graph, coloring by path length from central node
p = nx.single_source_shortest_path_length(G, ncenter)
plt.figure()
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist=list(p.keys()),
                       node_size=120, alpha=0.5,
                       node_color=list(p.values()),
                       cmap=plt.cm.jet_r)
plt.show()

#go to cmd windows and write jupyter notebook to start a new notebook

#show all installed packages
pip list

#list all magic commands
%lsmagic

#A Test Program
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

N = 20
θ = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, polar=True)
bars = ax.bar(θ, radii, width=width, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.5)

plt.show()

#in python to run a entire .py file write
# %run first.py

#present working directory, to see which working directory we are
pwd

#list all objects and files in file
ls

#asks Jupyter to print the contents of first.py, this is for bash
cat first.py

#asks Jupyter to print the contents of first.py in windows
type test.py
