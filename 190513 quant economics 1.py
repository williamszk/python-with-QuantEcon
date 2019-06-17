
# https://lectures.quantecon.org/py/about_py.html

############################################################################
#An Introductory Example ###################################################
############################################################################
# <editor-fold> An Introductory Example  **********************************
# An Introductory Example
# Exercise 4
from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
import numpy as np                     # Load the library
from scipy.stats import norm
from scipy.integrate import quad
from sympy import Symbol
from sympy import solve
from sympy import limit, sin, diff
import pandas as pd
import networkx as nx
import this
aux_1 = [uniform() for i in range(10)]
aux_2 = [int(i < .5) for i in aux_1]
val_1 = 0
val_2 = 0
for i in aux_2:
    if i == 1:
        val_1 = 1 + val_1
    if val_1 == 3:
        val_2 = 1
    if i == 0:
        val_1 = 0
print(val_2)
print(aux_2)

# solution from the book
payoff = 0
count = 0
for i in range(10):
    U = uniform()
    count = count + 1 if U < 0.5 else 0
    if count == 3:
        payoff = 1
print(payoff)


# Exercise 5
T = 200
alpha = 0.9
series = []
aux_1 = [np.random.randn() for i in range(T)]
val_1 = 0
for i in aux_1:
    val_1 = i + alpha*val_1
    series.append(val_1)
plt.plot(series)
plt.show()


# solution from the book
α = 0.9
ts_length = 200
current_x = 0
x_values = []
for i in range(ts_length + 1):
    x_values.append(current_x)
    current_x = α * current_x + np.random.randn()
plt.plot(x_values)
plt.show()


# Exercise 6


x = [np.random.randn() for i in range(100)]
plt.plot(x, label="white noise")
plt.legend()
plt.show()


alphas = [0, 0.8, 0.98]
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


# solution from the book
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
    plt.plot(x_values, label='α ='+str(α))
plt.legend()
plt.show()


# Setting up Your Python Environment
# Version 1
%matplotlib tk

x = np.random.randn(100)
plt.plot(x)
plt.show()


# this text show some examples of what python can do


a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
b = np.cos(a)                          # Apply cosine to each element of a
c = np.sin(a)                          # Apply sin to each element of a

b @ c  # take the inner product


ϕ = norm()
value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
value


# Symbolic Algebra
# It’s useful to be able to manipulate symbolic expressions, as in Mathematica or Maple

x, y = Symbol('x'), Symbol('y')  # Treat 'x' and 'y' as algebraic symbols
x + x + x + y

expression = (x + y)**2
expression.expand()

expr1 = (x + y)**5
expr1.expand()

z = Symbol('z')
expr2 = (x+y+z)**2
expr2.expand()

# solve polynomials
solve(x**2 + x + 2)

# calculate limits, derivatives and integrals
limit(1 / x, x, 0)
limit(1 / x, x, 1)

limit(sin(x) / x, x, 0)

diff(sin(x), x)

#Statistics ###################################################################
np.random.seed(1234)

data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
dates = pd.date_range('28/12/2010', periods=5)

df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
print(df)

df.mean()


#Networks and Graphs ##########################################################
# %matplotlib inline
%matplotlib tk
np.random.seed(1234)

# Generate random graph
p = dict((i, (np.random.uniform(0, 1), np.random.uniform(0, 1))) for i in range(200))
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

# go to cmd windows and write jupyter notebook to start a new notebook

# show all installed packages
pip list

# list all magic commands
%lsmagic

# A Test Program
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

# in python to run a entire .py file write
# %run first.py

# present working directory, to see which working directory we are
pwd

# list all objects and files in file
ls

# asks Jupyter to print the contents of first.py, this is for bash
cat first.py

# asks Jupyter to print the contents of first.py in windows
type test.py

# </editor-fold> ********************************************************************


############################################################################
#Python Essentials       ###################################################
############################################################################
# <editor-fold> Python Essentials  **********************************
# Python Essentials
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

# complex numbers
x = complex(1, 2)
y = complex(2, 1)
x * y


# Containers
# tuples
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
type(x)

# Python lists are mutable
x = [1, 2]
x[0] = 10
x

x = (1, 2)
x[0] = 10

# Tuples (and lists) can be “unpacked” as follows
integers = (10, 20, 30)
x, y, z = integers
x

list1 = [10, 20, 30]
type(list1)
x, y, z = list1

# Slice Notation
a = [2, 4, 6, 8, 10, 12, 14]
a[1:]  # 1 onwards
a[2:]  # 2 onwards
a[1:3]
a[1:5]
# The general rule is that a[m:n] returns n - m elements, starting at a[m]

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


# Sets are unordered collections without duplicates,
# and set methods provide the usual set theoretic operations
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

f = open('newfile.txt', 'r')  # 'r' means that we want to read the file
out = f.read()
out
print(out)

f = open('C:\\Users\\willi\\Desktop\\working\\projects_git\\python-with-QuantEcon\\newfile.txt', 'r')
out = f.read()
print(out)

# Iteraing

% % file us_cities.txt
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

x = ('a', 'b')  # tuple, immutable list
x

# Looping without Indices

x_values = [1, 2, 3]  # Some iterable x
for x in x_values:
    print(x * x)

for i in range(len(x_values)):
    print(x_values[i] * x_values[i])

countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')

names = ['Tom', 'John']
marks = ['E', 'F']
dict(zip(names, marks))

letter_list = ['a', 'b', 'c']
for index, letter in enumerate(letter_list):
    print(f"letter_list[{index}] = '{letter}'")

# Comparisons and Logical Operators
# Comparisons

x, y = 1, 2
x < y

1 < 2 < 3

x = 1    # Assignment
x == 2   # Comparison
1 != 2

x = 'yes' if 42 else 'no'
x

x = 'yes' if [] else 'no'
x

# Combining Expressions

1 < 2 and 'f' in 'foo'
1 < 2 and 'z' in 'foo'
1 < 2 or 'z' in 'foo'

not True

not not True

# More Functions
max(19, 20)

range(4)  # in python3 this returns a range iterator object

list(range(4))  # will evaluate the range iterator and create a list
list(range(10))

str(22)
type(22)

bools = False, True, True
all(bools)  # True if all are True and False otherwise
type(bools)
any(bools)  # False if all are False and True otherwise

# The Flexibility of Python Functions

# Python functions will output the first return that appear


def f(x):
    if x < 0:
        return 'negative'
    return 'nonnegative'


f(1)
f(-10)

# Docstrings


def f(x):
    """
    This function squares its argument
    """
    return x**2


f(10)
f?
f??

# One-Line Functions


def f(x):
    return x**3


f(3)


def f(x): return x**2


f(3)


quad(lambda x: x**3, 0, 2)

# Keyword Arguments
plt.plot(x, 'b-', label="white noise")


def f(x, a=1, b=1):
    return a + b * x


f(2)
f(2, a=4, b=5)

# Coding Style and PEP8


# Exercise 1
# part 1
x_vals, y_vals = (1, 2, 3), (4, 5, 6)
aux1 = 0
for i, j in zip(x_vals, y_vals):
    aux1 = i*j + aux1
aux1

# their solution for part 1
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
# usually is better to use List Comprehensions when you are creating lists with loops

# part 2
aux1 = [x % 2 == 0 for x in range(100)]
sum(aux1)
# their solution for part 2
# almost the same
sum([x % 2 == 0 for x in range(100)])

len([x for x in range(100) if x % 2 == 0])


# part 3
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
aux1 = [pair[0] % 2 == 0 and pair[1] % 2 == 0 for pair in pairs]
sum(aux1)

# their solution for part 3
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])

# Exercise 2


def p(x, coeff=[1, 1, 1]):
    aux1 = [jj*(x**ii) for ii, jj in enumerate(coeff)]
    return sum(aux1)


p(3)
# it is the same thing to use the following:
[jj*x ^ ii for ii, jj in zip(range(len(coeff)), coeff)]

p(1, (2, 4))

# Exercise 3
aux1 = 'foo'
aux1[0]
'foo'.upper()
aux1 = 'Write a Function that takes a string As an argument and Returns the Iumber of capital Letters in Lhe string'
aux1.upper()
'W' == 'w'
' ' == ' '.upper()
count1 = [ii == jj for ii, jj in zip(aux1, aux1.upper())]
count2 = [ii == ' ' for ii in aux1]
sum(count1) - sum(count2)

# their solution for exercise 3


def f(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count


f('The Rain in Spain')
f(aux1)

letter = ' '
letter.isalpha()
' '.isalpha()
'2'.isalpha()
'a'.isalpha()
'asdfasdf'.isalpha()


# Exercise 4
seq_a = [1, 2, 3]
seq_b = [1, 2, 3, 4, 3]


def matching1(seq_a, seq_b):
    aux2 = 0
    for ii in seq_a:
        aux1 = sum([ii == jj for jj in seq_b])
        if aux1 != 0:
            aux1 = 1
        aux2 = aux2 + aux1
    if aux2 == len(seq_a):
        return('seq_a is contained in seq_b')
    if aux2 != len(seq_a):
        return('seq_a is not contained in seq_b')
#


matching1(seq_a, seq_b)

# their solution for exercise 4
seq_a = [1, 2, 3]
seq_b = [1, 2, 3, 4, 3]


def f(seq_a, seq_b):
    is_subset = True
    for a in seq_a:
        if a not in seq_b:
            is_subset = False
    return is_subset
#


print(f([1, 2], [1, 2, 3]))
print(f([1, 2, 3], [1, 2]))

# Exercise 5
a = 0
b = 5
x = 2.123


def f(x):
    x**2


n = 10


def linapprox(f, a, b, n, x):
    aux_width = (b-a)/n
    aux1 = [ii*aux_width for ii in list(range(n))]
    aux1_2 = [ii+aux_width for ii in aux1]
    aux2 = [ii < x < jj for ii, jj in zip(aux1, aux1_2)]
    aux1[aux2]

# their solution for exercise 5


def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval
    [a, b], with n evenly spaced grid points.

    Parameters
    ===========
        f : function
            The function to approximate

        x, a, b : scalars (floats or integers)
            Evaluation point and endpoints, with a <= x <= b

        n : integer
            Number of grid points

    Returns
    =========
        A float. The interpolant evaluated at x

    """
    length_of_interval = b - a
    num_subintervals = n - 1
    step = length_of_interval / num_subintervals

    # === find first grid point larger than x === #
    point = a
    while point <= x:
        point += step

    # === x must lie between the gridpoints (point - step) and point === #
    u, v = point - step, point

    return f(u) + (x - u) * (f(v) - f(u)) / (v - u)

# </editor-fold> **********************************


############################################################################
# OOP I: Introduction to Object Oriented Programming #######################
############################################################################
# <editor-fold>   OOP I: Introduction to Object Oriented Programming **********************************

# Type
s = 'This is a string'
type(s)

x = 42   # Now let's create an integer
type(x)

'300' + 'cc'

300 + 400

int('300') + 400   # To add as numbers, change the string to an integer

# Identity
y = 2.5
z = 2.5
id(y)
id(z)

# Object Content: Data and Attributes
x = 42
x

x.imag  # the imaginary part of the number
x.__class__  # an attribute of type
type(x)

# Methods


# </editor-fold> ****************************************************************************************
