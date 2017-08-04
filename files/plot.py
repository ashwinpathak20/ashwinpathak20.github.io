from scipy import special
from scipy.optimize import rosen, rosen_der
import numpy as np
from _derivative_numdiff import derivative as derivative1
from _jacobian_numdiff import gradient as gradient1
import matplotlib.pyplot as plt


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

err = []
np.random.seed(0)
x = np.random.rand(5)*20 - 15

err.append(max(abs(derivative1(ai, x)-aip(x))))
err.append(max(abs(derivative1(bi, x)-bip(x))))
err.append(max(abs(derivative1(bie, x)-biep(x))))

np.random.seed(0)
y = np.random.rand(5)*10

err.append(max(abs(derivative1((lambda y: special.jv(1, y)), y)-special.jvp(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.yv(1, y)), y)-special.yvp(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.kv(1, y)), y)-special.kvp(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.iv(1, y)), y)-special.ivp(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.spherical_jn(1, y)), y)-special.spherical_jn(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.spherical_yn(1, y)), y)-special.spherical_yn(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.spherical_kn(1, y)), y)-special.spherical_kn(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.spherical_in(1, y)), y)-special.spherical_in(1, y, 1))))
err.append(max(abs(derivative1((lambda y: special.ber(y)), y)-special.berp(y))))
err.append(max(abs(derivative1((lambda y: special.bei(y)), y)-special.beip(y))))
err.append(max(abs(derivative1((lambda y: special.ker(y)), y)-special.kerp(y))))
err.append(max(abs(derivative1((lambda y: special.kei(y)), y)-special.keip(y))))
err.append(max(abs(np.ravel(gradient1(f, x))-df(x))))

ind = np.arange(16)
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(ind, err, width, color='r')
plt.yscale('log')
ax.set_ylabel('max(abs(err))')
ax.set_xticks(ind)
ax.set_xticklabels(('ai', 'bi', 'bie', 'jv', 'yv', 'kv',
                   'iv', 's_jn', 's_yn', 's_kn',
                   's_in', 'ber', 'bei', 'ker', 'kei', 'ros'))
plt.show()