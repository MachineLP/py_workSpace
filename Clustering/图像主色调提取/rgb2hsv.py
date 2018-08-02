# -*- coding: utf-8 -*-
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r and g >= b:
        h = 60 * ((g - b) / df) + 0
    elif mx == r and g < b:
        h = 60 * ((g-b)/df) + 360
    elif mx == g:
        h = 60 * ((b-r)/df) + 120
    elif mx == b:
        h = 60 * ((r-g)/df) + 240
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v
 
def rgb2hsv2(R, G, B):
    mx = max(R, G, B)
    mn = min(R, G, B)
    if R == mx:
        H = (G-B) / (mx-mn)
    elif G == mx:
        H = 2 + (B-R) / (mx-mn)
    elif B == mx:
        H = 4 + (R-G) / (mx-mn)
    H = H * 60
    if H < 0:
        H = H + 360
    V = mx
    S = (mx - mn) / mx
    return H, S, V

 
