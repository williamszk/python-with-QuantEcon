# <editor-fold> Numeric Python  **********************************
from numpy.random import uniform
from numpy import cumsum
from numpy import resize
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
%matplotlib tk

x = np.random.uniform(0, 1, size=1000000)
x.mean()

a = np.zeros(3)
a
type(a)
type(a[0])
a = np.zeros(3, dtype=int)
type(a[0])
z = np.zeros(10, dtype=int)
z = np.zeros(10)
z.shape
z.shape = (10, 1)
z.shape
z = np.zeros(4)
z.shape = (2, 2)
z = np.zeros((2, 2))

# Creating Arrays
z = np.empty(3)
z = np.linspace(2, 4, 5)
z = np.linspace(2, 4, 10)
z = np.identity(2)
z = np.eye(3)
w = [10, 20]
type(w)
z = np.array(w)
type(z)
w = (10, 20)
w[1]
type(w)
z = np.array(w, dtype=float)
type(z)
# list of lists
w = [[10, 20], [20, 30]]
z = np.array(w)
z.shape
w = [[10, 20], [30, 20], [40, 50]]
z = np.array(w)
z.shape
na = np.linspace(10, 20, 2)
na is np.asarray(na)
na2 = np.array(na)
na is na2
id(na)
id(na2)
# create a text file
pwd
w = [[10, 20], [30, 20], [40, 50]]
z = np.array(w)
z.shape

np.savetxt("teste1.txt", z)
z1 = np.loadtxt("teste1.txt")
type(z1)


# Array Indexing
z = np.linspace(1, 2, 5)
z

z[0]
z[0:2]
z[-1]
z = np.array([[1, 2], [3, 4]])
z
z[0, 0]
z[0, 1]

z[0, :]
z1 = z[:, 1]
z1
z1.shape
z1.shape = (2, 1)

z = np.linspace(2, 4, 5)
z

indices = np.array((0, 2, 3))
z[indices]
z[np.array([0, 2, 3])]

z

d = np.array([0, 1, 1, 0, 0], dtype=bool)
d
type(d)

z[d]

z = np.empty(3)
type(z)

z[:] = 42

# Array Methods
a = np.array((4, 3, 2, 1))
a
a.sort()
a
a.sum()
a.mean()
a.max()
a.argmax()
a.cumsum()
a.var()
a.std()
a.shape
a = np.array((4, 3, 2, 1))
a.shape = (2, 2)
a.T
z = np.linspace(2, 4, 5)
z.searchsorted(2.2)

a = np.array((4, 3, 2, 1))
np.sum(a)
a.sum()

np.mean(a)
a.mean()

# Operations on Arrays
# Arithmetic Operations¶

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
a + b

a*b
a+10
a*10

A = np.ones((2, 2))
B = np.ones((2, 2))
A+B
A+10
A*B

# Matrix Multiplication
A = np.ones((2, 2))
B = np.ones((2, 2))
A @ B
type(B)
C = np.zeros((2, 2))
A @ C
A = np.array((1, 2))
B = np.array((10, 20))
A @ B
A = np.array(((1, 2), (3, 4)))
A
A @ (0, 1)

# Mutability and Copying Arrays
a = np.array([42, 44])
a
a[-1] = 0
a
a = np.random.randn(3)
a
b = a
b
b[0] = 0.0
b
a

# Making Copies
a = np.random.randn(3)
a
b = np.copy(a)
b
b[:] = 1
b
a

# Additional Functionality
z = np.array([1, 2, 3])
z

np.sin(z)

n = len(z)
y = np.empty(n)
for i in range(n):
    y[i] = np.sin(z[i])
y
z
(1 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z**2)


def f(x):
    return 1 if x > 0 else 0


x = np.random.randn(4)

np.where(x > 0, 1, 0)


def f(x): return 1 if x > 0 else 0


f = np.vectorize(f)
f(x)
# now it accepts vectors inside

# Comparisons
z = np.array([2, 3])
y = np.array([2, 3])
z == y

y[0] = 5
z == y

z != y

z = np.array([2])
y = np.array([2, 3])
z == y

z = np.linspace(0, 10, 5)
z > 3

b = z > 3
z[b]
z[z > 3]

# Subpackages
z = np.random.randn(10000)  # Generate standard normals
y = np.random.binomial(10, 0.5, size=1000)    # 1,000 draws from Bin(10, 0.5)
y.mean()

A = np.array([[1, 2], [3, 4]])
# Compute the determinant
np.linalg.det(A)
np.linalg.inv(A)

np.linalg.inv(A) @ A

# </editor-fold>   **********************************

# <editor-fold> Exercises  **********************************

# Exercise 1
a = np.random.randn(100000)

np.cumprod()

a = np.random.randn(10)
coeff = a
x = 2


def p(x, coeff):
    coeff
    aux_1 = x**np.array(range(len(coeff)))
    aux_2 = aux_1*coeff
    return aux_2.sum()


results1 = p(3, coeff)

# Exercise 1 their solution
coef = np.random.randn(5)
x = 2


def p(x, coef):
    X = np.empty(len(coef))
    X[0] = 1
    X[1:] = x
    y = np.cumprod(X)   # y = [1, x, x**2,...]
    return coef @ y


coef = np.ones(3)
print(coef)
print(p(1, coef))
# For comparison
q = np.poly1d(coef)
print(q(1))

# Exercise 2

q = np.array([.4, .6])
q.sum()


def sample(q):
    a = 0.0
    U = uniform(0, 1)
    for i in range(len(q)):
        if a < U <= a + q[i]:
            return i
        a = a + q[i]


###
q = np.array([.4, .6])
q.sum()


def create_q():
    n = 4
    sections1 = [np.random.uniform(0, 1) for ii in range(n)]
    type(sections1)
    sec1 = np.array(sections1)
    type(sec1)
    len(sec1)
    sec1.sort()
    sec2 = sec1[1:len(sec1)]
    type(sec2)
    len(sec2)
    sec3 = np.insert(sec2, len(sec2), 1)
    len(sec3)
    sec4 = sec3-sec1
    sum(sec4)
    sec5 = np.insert(sec4, 0, sec1[0])
    sum(sec5)
    return sec5


q = create_q()


def sample_2(q):
    aux_q1 = np.cumsum(q)
    U = uniform(0, 1)
    pos1 = aux_q1.searchsorted(U)
    return pos1


sample_2(q)


# <editor-fold> an example to understand np.insert  ************
a = np.array([[1, 2], [3, 4], [5, 6]])
#'Axis parameter not passed. The input array is flattened before insertion.'
np.insert(a, 3, [11, 12])
#'Axis parameter passed. The values array is broadcast to match input array.'
# 'Broadcast along axis 0:'
np.insert(a, 1, [11], axis=0)
np.insert(a, 1, [11, 12], axis=0)
# '\n'

#'Broadcast along axis 1:'
np.insert(a, 1, [11, 12, 13], axis=1)
# </editor-fold> **********************************
# Hint: Use np.searchsorted and np.cumsum
z = np.linspace(2, 4, 5)
z.searchsorted(3.2)

# Exercise 3


# </editor-fold> Exercises  **********************************
