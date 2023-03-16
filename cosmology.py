# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:08:38 2023

@author: swebe
"""

from opt import gauss_newton, lm, j, f
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    data = np.genfromtxt('jla_mub.txt',dtype = float, names = ['z','mu'])
    
    x0 = np.array([50, 0.5]) #начальные приближения
    
    xx = np.linspace(0.01, 1.3, 100)
    rlm = lm(data['mu'], lambda *args: f(data['z'], *args),lambda *args: j(data['z'],*args), x0)
    
    x0 = np.array([50, 0.5])
    rgn = gauss_newton(data['mu'],lambda *args:f(data['z'],*args),lambda *args: j(data['z'],*args),x0)
    x0 = np.array([50, 0.5])
    
    plt.figure(figsize=(14, 10))
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.grid()
    plt.xlabel("Красное смещение", fontsize=17)
    plt.ylabel("Модуль расстояния", fontsize=19)
    
    plt.plot(xx ,f(xx, *rlm.x), lw=2,  color='indigo', label= ' Метод Левенберга—Марквардта')
    plt.plot(xx ,f(xx, *rgn.x),'--' , lw=1, color='plum', label = 'Метод Гаусса—Ньютона' )
    plt.plot(data['z'], data['mu'], '.', color='orange', ms= 7, label = 'Аппроксимируемые данные')
    plt.title('график зависимости μ от z', fontsize=25)
    plt.legend(loc='upper left' , fontsize='x-large')
    plt.savefig('mu-z.png')
    
    
    plt.figure(figsize=(8, 5))
    
    plt.xticks(range(rlm.nfev), fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.grid()
    plt.xlabel("итерационный шаг", fontsize='x-large')
    plt.ylabel("Функция потерь", fontsize='x-large')
    plt.plot(range(rlm.nfev),  rlm.cost, color='indigo', label=' Метод Левенберга—Марквардта')
    plt.plot(range(rgn.nfev), rgn.cost,'--', color='plum', label='Метод Гаусса—Ньютона')
    plt.title('зависимость функции потерь от итерационного шага', fontsize=14 )
    plt.legend()
    plt.savefig('cost.png')  
    
    
    d = {
        "Gauss-Newton": {"H0": round(rgn.x[0]), "Omega": round(rgn.x[1], 2), "nfev": rgn.nfev},
        "Levenberg-Marquardt": {"H0": round(rlm.x[0]), "Omega": round(rlm.x[1], 2), "nfev": rlm.nfev}
        }
    with open('parameters.json', 'w') as f:
        json.dump(d, f)
        
        
