#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:52:32 2025

@author: whadyimac
"""
import numpy as np

# Función a minimizar
def f(x):
    return (x - 3)**3 + 2

# Derivada de la función
def df(x):
    return 3*(x - 3)**2

#segunda derivada
def d2f(x):
    return 6*(x-3)

x0=4



#calculo del nuevo punto mas cercano al minimo
for i in range(5):
    s=-df(x0)*(d2f(x0))**(-1)
    x_nuevo=x0+s
    x0=x_nuevo
    print(x_nuevo,f(x_nuevo),df(x_nuevo),d2f(x_nuevo))















