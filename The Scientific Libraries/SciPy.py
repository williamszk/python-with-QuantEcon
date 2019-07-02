# <editor-fold> preliminaries  **********************************
# Import numpy symbols to scipy namespace
from scipy.optimize import fminbound
from scipy.optimize import fixed_point
from scipy.optimize import newton
from scipy.optimize import bisect
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import brentq
from scipy.integrate import quad
import numpy as np
from numpy.lib.scimath import *
from numpy.fft import fft, ifft
from numpy.random import rand, randn
from numpy import *
import numpy as _num
%matplotlib inline
# </editor-fold>   **********************************
# <editor-fold> SciPy versus NumPy  **********************************

linalg = None

__all__ = []
__all__ += _num.__all__
__all__ += ['randn', 'rand', 'fft', 'ifft']

del _num
# Remove the linalg imported from numpy so that the scipy.linalg package can be
# imported.
del linalg
__all__.remove('linalg')


a = np.identity(3)
print(a)

# </editor-fold>   **********************************
# <editor-fold> Statistics **********************************
np.random.beta(5, 5, size=3)

q = beta(5, 5)      # Beta(a, b), with a = b = 5
obs = q.rvs(2000)   # 2000 observations
obs.shape
grid = np.linspace(0.01, 0.99, 100)
grid.shape

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(obs, bins=40, density=True)
ax.plot(grid, q.pdf(grid), 'k-', linewidth=2)
plt.show()

q.cdf(0.4)
q.pdf(0.4)
q.ppf(0.8)

q.mean()

obs = beta.rvs(5, 5, size=2000)
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, beta.pdf(grid, 5, 5), 'k-', linewidth=2)
plt.show()

# Other Goodies in scipy.stats

x = np.random.randn(200)
y = 2 * x + 0.1 * np.random.randn(200)
gradient, intercept, r_value, p_value, std_err = linregress(x, y)
gradient, intercept

# </editor-fold>   *******************************
# <editor-fold> Roots and Fixed Points  **********************************


def f(x): return np.sin(4 * (x - 1/4)) + x + x**20 - 1


x = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))
plt.plot(x, f(x))
plt.axhline(ls='--', c='k')
plt.show()

# finding roots of a function using the bisection technique


def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b

    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        # === if root is between lower and middle === #
        if f(middle) > 0:
            lower, upper = lower, middle
        # === if root is between middle and upper  === #
        else:
            lower, upper = middle, upper

    return 0.5 * (upper + lower)


bisect(f, 0, 1)

# </editor-fold>   *******************************
# <editor-fold> The Newton-Raphson Method **********************************
#from scipy.optimize import newton
newton(f, 0.2)   # Start the search at initial condition x = 0.2
newton(f, 0.7)   # Start the search at x = 0.7 instead

%timeit bisect(f, 0, 1)
%timeit newton(f, 0.2)

# Hybrid Methods
brentq(f, 0, 1)
%timeit brentq(f, 0, 1)

# Multivariate Root-Finding
#from scipy.optimize import fixed_point
fixed_point(lambda x: x**2, 10.0)  # 10.0 is an initial guess

# Optimization
#from scipy.optimize import fminbound
fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]
# Multivariate Optimization
# </editor-fold>   *******************************
# <editor-fold> Integration **********************************
#from scipy.integrate import quad
integral, error = quad(lambda x: x**2, 0, 1)
integral


# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
# <editor-fold> **********************************
# </editor-fold>   *******************************
