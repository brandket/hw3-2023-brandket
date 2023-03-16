#!/usr/bin/env python3


from collections import namedtuple
import numpy as np
import scipy.integrate as intg


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

nm = lambda x : np.inner(x, x)

def f(z,H0,w):
    intgrf = lambda x: 1/(np.sqrt((1-w)*(1+x)**3 + w)) #подынтегральная функция
    d = lambda x1: 3e11/H0 *(1+x1)*intg.quad(intgrf,0,x1)[0]
    f =  5 * np.log10(np.array(list([d(x) for x in z]))) - 5 #модуль расстояния mu
    return f

# np.log(d) = np.log10(d) * np.log(10), т.е. mu = 5 * np.log(d)/np.log(10) - 5
def j(z,H0,w):
    j = np.zeros((z.shape[0], 2)) #Создаю матрицу Якобиана
    j[:,0] = -5/(H0* np.log(10)) #частная производная по H0
    
    """далее считаю для второго столбца значения, 
    т.е. производную по w. d(Mu)/dw = 5/np.log(10) * d(np.log(const * f(w)))/dw 
    = 5/np.log(10) * f'(w)/f(w)"""
    intgrf = lambda x: 1/(np.sqrt((1-w)*(1+x)**3 + w))
    denomint = lambda x2: intg.quad(intgrf,0,x2)[0] 
    denom = np.array(list([denomint(x) for x in z])) #знаменатель f(w)
    
    numerk = lambda x3: ((1 + x3)**3 - 1)/((2*((1-w)*(1+x3)**3 + w)**(3/2))) 
    numerint = lambda q3: intg.quad(numerk,0,q3)[0]  #числитель f'(w) 
    
    j[:,1] = 5/np.log(10) * np.divide(np.array([ numerint(x) for x in z]), denom)
    return j




#Далее идет код с формулами с лекций

def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    x = np.asarray(x0, dtype = float)
    i = 0
    cost = []
    while True:
        i += 1
        jac = j(*x)
        r = f(*x)-y
        cost.append(0.5 * nm(r))
        grad = jac.T @ r
        gradn = np.sqrt(nm(grad))
        delta = np.linalg.solve(nm(jac.T), -grad)
        x = x + k * delta
        if i <= 2:
            continue
        if np.abs(cost[-2] - cost[-1]) / cost[-1] <= tol:
            break
    return Result(nfev = i, cost = cost, gradnorm = gradn, x = x)


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    while True:
        i += 1
        jac = j(*x)
        r = f(*x) - y
        cost.append(0.5 * nm(r))
        grad = jac.T @ r
        gradn = np.sqrt(nm(grad))
        a = nm(jac.T)
        delta_l = np.linalg.solve(lmbd0 * np.eye(a.shape[0]) + a, - grad)
        f_l = 0.5 * nm(f(*(x + delta_l)) - y)
        delta_nu = np.linalg.solve(lmbd0 * np.eye(a.shape[0])/ nu + a, - grad)
        f_nu = 0.5 * nm(f(*(x + delta_nu)) - y)
        if f_nu <= cost[-1]:
            lmbd0 = lmbd0 / nu
            x += delta_nu
        elif (f_nu > cost[-1]) and (f_l <= cost[-1]):
            x += delta_l
        else:
            n = nu ** (1 / (nu*10))
            lmbd0 = lmbd0 * n
            delta_n = np.linalg.solve(lmbd0 * np.eye(a.shape[0]) + a, - grad)
            f_n = 0.5 * nm(f(*(x + delta_n)) - y)
            while f_n > cost[-1]:
                lmbd0 = lmbd0 * n
                delta_n = np.linalg.solve(lmbd0 * np.eye(a.shape[0]) + a, - grad)
                f_n = 0.5 * nm(f(*(x + delta_n)) - y)
            x = x + delta_n
        if i <= 2:
            continue
        if np.abs(cost[-1] - cost[-2]) <= tol * cost[-1]:
            break
    return Result(nfev = i, cost = cost, gradnorm = gradn, x = x)
 
    
 
    
 
   