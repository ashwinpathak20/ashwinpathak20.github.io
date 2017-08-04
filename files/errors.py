from scipy import special
from scipy.optimize import rosen, rosen_der, leastsq, curve_fit
import numpy as np
from _derivative_numdiff import derivative as derivative1
from _jacobian_numdiff import gradient as gradient1
import matplotlib.pyplot as plt

c = 1e+10
d = 1e-4

def ai(x):
    f = special.airy(x)
    return f[0]

def aip(x):
    f = special.airy(x)
    return f[1]

def bi(x):
    f = special.airy(x)
    return f[2]

def bip(x):
    f = special.airy(x)
    return f[3]

def bie(x):
    f = special.airye(x)
    return f[2]

def biep(x):
    f = special.airye(x)
    return f[3]

def f(x):
    return rosen(x)

def df(x):
    return rosen_der(x)

def f1(x):
    return np.exp(c*x)

def df1(x):
    return c*np.exp(c*x)

def f2(x):
    return d*np.exp(-2*x/d) + 1/d**2

def df2(x):
    return -2*np.exp(-2*x/d)

np.random.seed(0)
x = np.random.rand(5)*20 - 15

err1 = abs(derivative1(ai, x)-aip(x))
err2 = abs(derivative1(bi, x)-bip(x))
err3 = abs(derivative1(bie, x)-biep(x))

np.random.seed(0)
y = np.random.rand(5)*10

err4 = abs(derivative1((lambda y: special.jv(1, y)), y)-special.jvp(1, y, 1))
err5 = abs(derivative1((lambda y: special.yv(1, y)), y)-special.yvp(1, y, 1))
err6 = abs(derivative1((lambda y: special.kv(1, y)), y)-special.kvp(1, y, 1))
err7 = abs(derivative1((lambda y: special.iv(1, y)), y, num_steps=11)-special.ivp(1, y, 1))
err8 = abs(derivative1((lambda y: special.spherical_jn(1, y)), y)-special.spherical_jn(1, y, 1))
err9 = abs(derivative1((lambda y: special.spherical_yn(1, y)), y)-special.spherical_yn(1, y, 1))
err10 = abs(derivative1((lambda y: special.spherical_kn(1, y)), y)-special.spherical_kn(1, y, 1))
err11 = abs(derivative1((lambda y: special.spherical_in(1, y)), y)-special.spherical_in(1, y, 1))
err12 = abs(derivative1((lambda y: special.ber(y)), y)-special.berp(y))
err13 = abs(derivative1((lambda y: special.bei(y)), y)-special.beip(y))
err14 = abs(derivative1((lambda y: special.ker(y)), y)-special.kerp(y))
err15 = abs(derivative1((lambda y: special.kei(y)), y)-special.keip(y))
err16 = abs(np.ravel(gradient1(f, x))-df(x))
#err17 = derivative1(f1, -1) - df1(-1)

np.random.seed(0)
y = np.random.rand(5)*5 - 5
#err18 = abs(derivative1(f2, y) - df2(y))

np.random.seed(0)
y = np.random.rand(5)*0.01 - 0.01
#err19 = abs(derivative1(f2, y) - df2(y))

def func(x, y):
    return x**2 + y**2

popt, pcov =  curve_fit(func, [1,2], [2,3])
print pcov