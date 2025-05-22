#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:07:01 2025

@author: whadyimac
"""
from numpy.linalg import solve
import numpy as np

Ab=np.array([[1,1,0],[2,3,1],[1,1,2]])
B=np.array([8,12,10])
Anb=np.array([[2,1],[1,0],[1,1]])

Bx2=np.array([2,1,1])
Bx5=([1,0,1])

Xb=solve(Ab,B)
Cx2=solve(Ab,Bx2)
Cx5=solve(Ab,Bx5)

