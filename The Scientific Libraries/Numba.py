# <editor-fold> preliminaries  **********************************
from numba import vectorize
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import quantecon as qe
import numpy as np
import random
%matplotlib inline
# </editor-fold>   **********************************
# <editor-fold> dynamic typing  **********************************

a, b = 10, 10
a + b

a, b = 'foo', 'bar'
a + b

a, b = ['foo'], ['bar']
a + b

qe.util.tic()   # Start timing
n = 100_000
sum = 0
for i in range(n):
    x = random.uniform(0, 1)
    sum += x**2
qe.util.toc()   # End timing

qe.util.tic()
n = 100_000
x = np.random.uniform(0, 1, n)
np.sum(x**2)
qe.util.toc()

# </editor-fold>   **********************************
# <editor-fold> Universal Functions  **********************************
# ufunc
np.cos(1.0)
np.cos(np.linspace(0, 5, 10))


def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)


xgrid = np.linspace(-3, 3, 50)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.7,
                linewidth=0.25)
ax.set_zlim(-0.5, 1.0)
plt.show()

# non-vectorized version


def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)


grid = np.linspace(-3, 3, 1000)
m = -np.inf

qe.tic()
for x in grid:
    for y in grid:
        z = f(x, y)
        if z > m:
            m = z

qe.toc()

# vectorized version


def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)


grid = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(grid, grid)

qe.tic()
np.max(f(x, y))
qe.toc()

# an example


def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
        x[t+1] = 4 * x[t] * (1 - x[t])
    return x


qm(5, 10)

x = qm(0.1, 250)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, 'b-', lw=2, alpha=0.8)
ax.set_xlabel('time', fontsize=16)
plt.show()

# speed up using numba
#from numba import jit

qm_numba = jit(qm)  # qm_numba is now a 'compiled' version of qm

qe.util.tic()
qm(0.1, int(10**5))
time1 = qe.util.toc()

qe.util.tic()
qm_numba(0.1, int(10**5))
time2 = qe.util.toc()

time1 / time2  # Calculate speed gain

# to create a numbafied version of a function just usu @jit before creating a Functions
@jit
def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
        x[t+1] = 4 * x[t] * (1 - x[t])
    return x


# </editor-fold>   **********************************
# <editor-fold> A Gotcha: Global Variables  **********************************
a = 1


@jit
def add_x(x):
    return a + x


print(add_x(10))

a = 2

print(add_x(10))


# Numba for Vectorization
#from numba import vectorize

@vectorize
def f_vec(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)


grid = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(grid, grid)

np.max(f_vec(x, y))  # Run once to compile

qe.tic()
np.max(f_vec(x, y))
qe.toc()


@vectorize('float64(float64, float64)', target='parallel')
def f_vec(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)


np.max(f_vec(x, y))  # Run once to compile

qe.tic()
np.max(f_vec(x, y))
qe.toc()


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
# <editor-fold> preliminaries  **********************************


# </editor-fold>   **********************************
