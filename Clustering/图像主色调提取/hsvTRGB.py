# -*- coding: utf-8 -*-
import math
 
def Hsv2Rgb(H, S, V):
    H /= 60.0  # sector 0 to 5
    i = math.floor(H)
    f = H - i  # factorial part of h
    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))
    if i == 0:
        R = V
        G = t
        B = p
    elif i == 1:
        R = q
        G = V
        B = p
    elif i == 2:
        R = p
        G = V
        B = t
    elif i == 3:
        R = p
        G = q
        B = V
    elif i == 4:
        R = t
        G = p
        B = V
    else:
        R = V
        G = p
        B = q
    return R*255, G*255, B*255

 
