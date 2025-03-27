import numpy as np
import pandas as pd

# functionizing the hand-drawn curves for Ce of a partially-contracted v-notch weir

# Kulin and Compton 1975 Fig 4-5: Ce for partially-contracted v-notch weirs
# Using GeoGebra on image, making points, then PolyFit() polynomial curve fitting
# P/B = 1.0
# 380.3109110700038x^(6) - 564.6042445931643x^(5) + 334.8942428178557x^(4) - 100.3409026491789x^(3) + 15.9645346761014x^(2) - 1.2744715661067x + 0.6181601989942
# P/B = 0.8
# -2098.188266193364x^(8) + 4822.136683514677x^(7) - 4618.7836909284715x^(6) + 2388.1739432881486x^(5) - 722.6932711275226x^(4) + 130.3406886569203x^(3) - 13.6548795540878x^(2) + 0.7652761902615x + 0.5604543922725
# P/B = 0.6
# 3.5509969570713x^(6) - 7.6054084567142x^(5) + 6.1075340060361x^(4) - 2.1470998115822x^(3) + 0.3240549526796x^(2) - 0.0118348270028x + 0.5777004528603
# P/B = 0.4
# -326.2748669937048x^(11) + 1999.9815196019158x^(10) - 5376.586364087006x^(9) + 8329.485298705671x^(8) - 8219.396721238372x^(7) + 5392.87319867412x^(6) - 2385.9814006935103x^(5) + 707.7160459825956x^(4) - 137.2078078095456x^(3) + 16.4747600800621x^(2) - 1.0931938246613x + 0.608460034015
# P/B = 0.3
# -55.5246831179059x^(11) + 385.9421994215697x^(10) - 1172.777992197495x^(9) + 2045.95965628624x^(8) - 2263.169149753214x^(7) + 1655.456934299524x^(6) - 811.1922174539758x^(5) + 264.4084043205384x^(4) - 55.8169137604434x^(3) + 7.2148793901404x^(2) - 0.5061745895971x + 0.5927399876207
# P/B = 0.2
# -0.1776021827636x^(8) + 1.2367993513345x^(7) - 3.3601376637499x^(6) + 4.6649758699716x^(5) - 3.5963305902167x^(4) + 1.5750016012957x^(3) - 0.3915495246612x^(2) + 0.0537761462335x + 0.5754430976103
# P/B = 0.1
# 0.0040400722952x^(4) - 0.0022811100135x^(3) - 0.0098778444155x^(2) + 0.0074465337219x + 0.5775820332236

# limit points: order of x-min, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1
# A=(0.0989811416743,0.5782130563246) # all lines
# B=(0.3898705692441,0.5889729523452) # P/B = 1.0
# C=(0.5037674165547,0.5927945381178) # P/B = 0.8
# D=(0.6562955781038,0.5933648839001) # P/B = 0.6
# E=(0.9959266153994,0.598745504488) # P/B = 0.4
# F=(1.1639322663365,0.5928052993589) # P/B = 0.3
# G=(1.1799090295757,0.5826789714125) # P/B = 0.2
# H=(1.1840280388483,0.5765773476658) # P/B = 0.1

# when interpolating lines of P/B, extrapolate and use last (x,y) plus some sloped linear middle line
def Ce_part_vnotch(H,P,B): # H is head above crest (invert of weir), P is vertical distance crest to upstream bed, B is width of approach channel (e.g., top width of entire weir, including any horizontal crest on either side of v-notch)
    HP = H/P
    PB = np.round(P/B, 2)
    # if H/B > 0.4:
    #     print('Warning: H/B = {:.3f} > 0.4 - this weir does not meet the criteria for partial contraction.'.format(H/B))
    # print('PB = ', PB)
    if PB > 0.09 and PB < 0.11:
        # if HP < 0.1 or HP > 1.18:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.18:
            Ce = 0.58
        else:
            Ce = 0.0040400722952*HP**4 - 0.0022811100135*HP**3 - 0.0098778444155*HP**2 + 0.0074465337219*HP + 0.5775820332236
    elif PB >= 0.11 and PB <=0.19:
        # if HP < 0.1 or HP > 1.18:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.18:
            Ce = 0.58
        else: # interpolate
            Ce_PB01 = 0.0040400722952*HP**4 - 0.0022811100135*HP**3 - 0.0098778444155*HP**2 + 0.0074465337219*HP + 0.5775820332236
            Ce_PB02 = -0.1776021827636*HP**8 + 1.2367993513345*HP**7 - 3.3601376637499*HP**6 + 4.6649758699716*HP**5 - 3.5963305902167*HP**4 + 1.5750016012957*HP**3 - 0.3915495246612*HP**2 + 0.0537761462335*HP + 0.5754430976103
            Ce = (Ce_PB01+Ce_PB02)/2
    elif PB > 0.19 and PB < 0.21:
        if HP < 0.1 or HP > 1.18:
            Ce = np.nan
            # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.18:
            Ce = 0.58
        else:
            Ce = -0.1776021827636*HP**8 + 1.2367993513345*HP**7 - 3.3601376637499*HP**6 + 4.6649758699716*HP**5 - 3.5963305902167*HP**4 + 1.5750016012957*HP**3 - 0.3915495246612*HP**2 + 0.0537761462335*HP + 0.5754430976103
    elif PB >= 0.21 and PB <=0.29:
        # if HP < 0.1 or HP > 1.16:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.16:
            Ce = 0.59
        else: # interpolate
            Ce_PB02 = -0.1776021827636*HP**8 + 1.2367993513345*HP**7 - 3.3601376637499*HP**6 + 4.6649758699716*HP**5 - 3.5963305902167*HP**4 + 1.5750016012957*HP**3 - 0.3915495246612*HP**2 + 0.0537761462335*HP + 0.5754430976103
            Ce_PB03 = -55.5246831179059*HP**11 + 385.9421994215697*HP**10 - 1172.777992197495*HP**9 + 2045.95965628624*HP**8 - 2263.169149753214*HP**7 + 1655.456934299524*HP**6 - 811.1922174539758*HP**5 + 264.4084043205384*HP**4 - 55.8169137604434*HP**3 + 7.2148793901404*HP**2 - 0.5061745895971*HP + 0.5927399876207
            Ce = (Ce_PB02+Ce_PB03)/2
    elif PB > 0.29 and PB < 0.31:
        # if HP < 0.1 or HP > 1.16:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.16:
            Ce = 0.59
        else:
            Ce = -55.5246831179059*HP**11 + 385.9421994215697*HP**10 - 1172.777992197495*HP**9 + 2045.95965628624*HP**8 - 2263.169149753214*HP**7 + 1655.456934299524*HP**6 - 811.1922174539758*HP**5 + 264.4084043205384*HP**4 - 55.8169137604434*HP**3 + 7.2148793901404*HP**2 - 0.5061745895971*HP + 0.5927399876207
    elif PB >= 0.31 and PB <=0.39:
        # if HP < 0.1 or HP > 1.1:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.1:
            Ce = 0.59
        elif HP > 1.0 and HP <= 1.1: # extrapolate
            Ce = 0.5933056322910634 + 0.05*(HP-1.0) # 0.05 just estimated from plotting and checking
        else: # interpolate
            Ce_PB03 = -55.5246831179059*HP**11 + 385.9421994215697*HP**10 - 1172.777992197495*HP**9 + 2045.95965628624*HP**8 - 2263.169149753214*HP**7 + 1655.456934299524*HP**6 - 811.1922174539758*HP**5 + 264.4084043205384*HP**4 - 55.8169137604434*HP**3 + 7.2148793901404*HP**2 - 0.5061745895971*HP + 0.5927399876207
            Ce_PB04 = -326.2748669937048*HP**11 + 1999.9815196019158*HP**10 - 5376.586364087006*HP**9 + 8329.485298705671*HP**8 - 8219.396721238372*HP**7 + 5392.87319867412*HP**6 - 2385.9814006935103*HP**5 + 707.7160459825956*HP**4 - 137.2078078095456*HP**3 + 16.4747600800621*HP**2 - 1.0931938246613*HP + 0.608460034015
            Ce = (Ce_PB03+Ce_PB04)/2
    elif PB > 0.39 and PB < 0.41:
        # if HP < 0.1 or HP > 1.0:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 1.0:
            Ce = 0.60
        else:
            Ce = -326.2748669937048*HP**11 + 1999.9815196019158*HP**10 - 5376.586364087006*HP**9 + 8329.485298705671*HP**8 - 8219.396721238372*HP**7 + 5392.87319867412*HP**6 - 2385.9814006935103*HP**5 + 707.7160459825956*HP**4 - 137.2078078095456*HP**3 + 16.4747600800621*HP**2 - 1.0931938246613*HP + 0.608460034015
    elif PB >= 0.41 and PB <=0.59:
        # if HP < 0.1 or HP > 0.8:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.8:
            Ce = 0.59
        elif HP > 0.66 and HP <= 0.8: # extrapolate 
            Ce = 0.5883652546103659 + 0.06*(HP-0.66) # 0.06 just estimated from plotting and checking
        else: # interpolate
            Ce_PB04 = -326.2748669937048*HP**11 + 1999.9815196019158*HP**10 - 5376.586364087006*HP**9 + 8329.485298705671*HP**8 - 8219.396721238372*HP**7 + 5392.87319867412*HP**6 - 2385.9814006935103*HP**5 + 707.7160459825956*HP**4 - 137.2078078095456*HP**3 + 16.4747600800621*HP**2 - 1.0931938246613*HP + 0.608460034015
            Ce_PB06 = 3.5509969570713*HP**6 - 7.6054084567142*HP**5 + 6.1075340060361*HP**4 - 2.1470998115822*HP**3 + 0.3240549526796*HP**2 - 0.0118348270028*HP + 0.5777004528603
            Ce = (Ce_PB04+Ce_PB06)/2
    elif PB > 0.59 and PB < 0.61:
        # if HP < 0.1 or HP > 0.66:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.66:
            Ce = 0.59
        else:
            Ce = 3.5509969570713*HP**6 - 7.6054084567142*HP**5 + 6.1075340060361*HP**4 - 2.1470998115822*HP**3 + 0.3240549526796*HP**2 - 0.0118348270028*HP + 0.5777004528603
    elif PB >=0.61 and PB <=0.79:
        # if HP < 0.1 or HP > 0.58:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.58:
            Ce = 0.59
        elif HP > 0.5 and HP <= 0.58: # extrapolate 
            Ce = 0.5882051036521176 + 0.05*(HP-0.5) # 0.07 just estimated from plotting and checking
        else: # interpolate
            Ce_PB06 = 3.5509969570713*HP**6 - 7.6054084567142*HP**5 + 6.1075340060361*HP**4 - 2.1470998115822*HP**3 + 0.3240549526796*HP**2 - 0.0118348270028*HP + 0.5777004528603
            Ce_PB08 = -2098.188266193364*HP**8 + 4822.136683514677*HP**7 - 4618.7836909284715*HP**6 + 2388.1739432881486*HP**5 - 722.6932711275226*HP**4 + 130.3406886569203*HP**3 - 13.6548795540878*HP**2 + 0.7652761902615*HP + 0.5604543922725
            Ce = (Ce_PB06+Ce_PB08)/2
    elif PB > 0.79 and PB < 0.81:
        # if HP < 0.1 or HP > 0.5:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.5:
            Ce = 0.59
        else:
            Ce = -2098.188266193364*HP**8 + 4822.136683514677*HP**7 - 4618.7836909284715*HP**6 + 2388.1739432881486*HP**5 - 722.6932711275226*HP**4 + 130.3406886569203*HP**3 - 13.6548795540878*HP**2 + 0.7652761902615*HP + 0.5604543922725
    elif PB >=0.81 and PB <=0.99:
        # if HP < 0.1 or HP > 0.45:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.45:
            Ce = 0.59
        elif HP > 0.39 and HP <= 0.45: # extrapolate 
            Ce = 0.5864044129015097 + 0.1*(HP-0.39) # 0.1 just estimated from plotting and checking
        else: # interpolate
            Ce_PB08 = -2098.188266193364*HP**8 + 4822.136683514677*HP**7 - 4618.7836909284715*HP**6 + 2388.1739432881486*HP**5 - 722.6932711275226*HP**4 + 130.3406886569203*HP**3 - 13.6548795540878*HP**2 + 0.7652761902615*HP + 0.5604543922725
            Ce_PB10 = 380.3109110700038*HP**6 - 564.6042445931643*HP**5 + 334.8942428178557*HP**4 - 100.3409026491789*HP**3 + 15.9645346761014*HP**2 - 1.2744715661067*HP + 0.6181601989942
            Ce = (Ce_PB08+Ce_PB10)/2
    elif PB > 0.99 and PB < 1.1:
        # if HP < 0.1 or HP > 0.39:
        #     Ce = np.nan
        #     # print('Error: H/P value out of bounds of plotted range.')
        if HP < 0.1:
            Ce = 0.58
        elif HP > 0.39:
            Ce = 0.59
        else:
            Ce = 380.3109110700038*HP**6 - 564.6042445931643*HP**5 + 334.8942428178557*HP**4 - 100.3409026491789*HP**3 + 15.9645346761014*HP**2 - 1.2744715661067*HP + 0.6181601989942
    else: 
        Ce = np.nan
        # print('Error: P/B value out of bounds of plotted range.')
    
    return Ce


# Q functions for weirs

def kh_vnotch_KC(angle_vnotch): # from Kulin and Compton 1975 - Fig 4-3
    return 0.0000000002853*angle_vnotch**4 - 0.0000000830929*angle_vnotch**3 + 0.0000097540537*angle_vnotch**2 - 0.0005756285132*angle_vnotch + 0.0174729907833

def Q_Kindsvater_Shen(angle_vnotch,h1,P,B): # from Kulin and Compton 1975 - Eq 4.4
    Ce = Ce_part_vnotch(h1,P,B)
    kh = kh_vnotch_KC(angle_vnotch)
    h1e = h1 + kh # head on weir, h1, plus head correction factor, kh
    if h1 <= 0: # if negative head means below weir crest, make flowrate zero
        Q = 0
    else:
        Q = 4.28*Ce*np.tan(angle_vnotch/2)*h1e**(5/2)
    return Q

def Ce_Phil(H): # uses Phil's jerry-rigged Excel polynomial curve fitting to find Ce coefficient
    return 0.0142*H**3 - 0.021*H**2 + 0.01*H + 0.5761



def Q_weir_Phil_frankenstein(P_vnotch,P_rect,Cw,L_rect,H): # uses Phil's frankenstein of rectangular and v-notch sharp-crested weir equations
    if H <= 0:
        Q_tot = 0
    elif H> 0 and H <= (P_rect-P_vnotch): 
        Ce = Ce_Phil(H)
        Q_tot = 4.28*Ce*(H+0.0029)**(5/2)
    else: # combined rectangular and vnotch
        Ce = Ce_Phil(H)
        Q_vnotch = 4.28*Ce*(H+0.0029)**(5/2)
        H_rect = H - (P_rect - P_vnotch)
        Q_rect = Cw*L_rect*H_rect**1.5
        Q_tot = Q_vnotch+Q_rect
    return Q_tot

# def Q_weir_Phil_frankenstein_alt1(P_vnotch,P_rect,Cw,L_rect,H): # uses Phil's frankenstein of rectangular and v-notch sharp-crested weir equations
#     if H <= 0:
#         Q_tot = 0
#     elif H> 0 and H <= (P_rect-P_vnotch): 
#         Q_tot = Q_Kindsvater_Shen(90,H,P_vnotch,4)
#     else: # combined rectangular and vnotch
#         Q_vnotch = Q_Kindsvater_Shen(90,H,P_vnotch,4)
#         H_rect = H - (P_rect - P_vnotch)
#         Q_rect = Cw*L_rect*H_rect**1.5
#         Q_tot = Q_vnotch+Q_rect
#     return Q_tot

def Q_Bergmann(hm,P_rect,P_vnotch,units_str,b): # Bergmann 1963 from USBR Water Msmt Manual 1997 and Quebec Ministry of the Environment 2007 - for combination v-notch/rectangular weirs - might be only for fully contracted v-notch weirs?
    # Q = C*ht**1.72 - d + e*b*hr # original Bergmann Eq - collapses to Cone Eq at H <=1 ft, just v-notch, hr=hm-ht
    ht = (P_rect-P_vnotch)
    hr = hm-ht
    if units_str == 'Imperial':
        C = 3.9
        d = 1.5
        e = 3.3
    elif units_str == 'SI':
        C = 5.2
        d = 1.5
        e = 1.82
    else:
        print('Error: no units system specified.')
    if hm <= 0:
        Q_tot = 0
    else:
        Q_tot = C*ht**1.72 - d + e*b*hr
    return Q_tot

def Q_Sam_frankenstein(hm,P_rect,P_vnotch,B,angle_vnotch,units_str,b): # Sam's frankenstein - combines Kindsvater-Shen for when flows only in v-notch, then Bergmann 1963 combined equation when above rectanglular crest
    # Q = C*ht**1.72 - d + e*b*hr # original Bergmann Eq - collapses to Cone Eq at H <=1 ft, just v-notch, hr=hm-ht
    if hm <= 0:
        Q_tot = 0
    elif hm > 0 and hm <= (P_rect-P_vnotch):
        Q_tot = Q_Kindsvater_Shen(angle_vnotch,hm,P_vnotch,B)
    else: # Bergmann 1963 from USBR Water Msmt Manual 1997 and Quebec Ministry of the Environment 2007 - for combination v-notch/rectangular weirs - might be only for fully contracted v-notch weirs?
        # Q_tot = Q_Bergmann(hm,P_rect,P_vnotch,units_str,b)
        Q_vnotch = Q_Kindsvater_Shen(angle_vnotch,(P_rect-P_vnotch),P_vnotch,B)
        ht = (P_rect-P_vnotch)
        hr = hm-ht
        if units_str == 'Imperial':
            C = 3.9
            d = 1.5
            e = 3.3
        elif units_str == 'SI':
            C = 5.2
            d = 1.5
            e = 1.82
        else:
            print('Error: no units system specified.')
        Q_tot = Q_vnotch + e*b*hr
    return Q_tot

# def Q_Sam_frankenstein_alt1(hm,P_rect,P_vnotch,B,angle_vnotch,units_str,b): # Sam's frankenstein - combines Kindsvater-Shen for when flows only in v-notch, then Bergmann 1963 combined equation when above rectanglular crest
#     # Q = C*ht**1.72 - d + e*b*hr # original Bergmann Eq - collapses to Cone Eq at H <=1 ft, just v-notch, hr=hm-ht
#     if hm <= 0:
#         Q_tot = 0
#     elif hm > 0 and hm <= (P_rect-P_vnotch):
#         Q_tot = Q_Kindsvater_Shen(angle_vnotch,hm,P_vnotch,B)
#     else: # Bergmann 1963 from USBR Water Msmt Manual 1997 and Quebec Ministry of the Environment 2007 - for combination v-notch/rectangular weirs - might be only for fully contracted v-notch weirs?
#         Q_tot = Q_Bergmann(hm,P_rect,P_vnotch,units_str,b)
#     return Q_tot

# def Q_Sam_frankenstein_alt2(hm,P_rect,P_vnotch,B,angle_vnotch,units_str,b): # Sam's frankenstein - combines Kindsvater-Shen for when flows only in v-notch, then Bergmann 1963 combined equation when above rectanglular crest
#     # Q = C*ht**1.72 - d + e*b*hr # original Bergmann Eq - collapses to Cone Eq at H <=1 ft, just v-notch, hr=hm-ht
#     if hm <= 0:
#         Q_tot = 0
#     elif hm > 0 and hm <= (P_rect-P_vnotch):
#         Q_tot = Q_Kindsvater_Shen(angle_vnotch,hm,P_vnotch,B)
#     else: # Bergmann 1963 from USBR Water Msmt Manual 1997 and Quebec Ministry of the Environment 2007 - for combination v-notch/rectangular weirs - might be only for fully contracted v-notch weirs?
#         # Q_tot = Q_Bergmann(hm,P_rect,P_vnotch,units_str,b)
#         Q_vnotch = Q_Kindsvater_Shen(angle_vnotch,(P_rect-P_vnotch),P_vnotch,B)
#         ht = (P_rect-P_vnotch)
#         hr = hm-ht
#         if units_str == 'Imperial':
#             C = 3.9
#             d = 1.5
#             e = 3.3
#         elif units_str == 'SI':
#             C = 5.2
#             d = 1.5
#             e = 1.82
#         else:
#             print('Error: no units system specified.')
#         Q_tot = Q_vnotch - d + e*b*hr
#     return Q_tot

# def Q_Sam_frankenstein_alt2(hm,P_rect,P_vnotch,B,angle_vnotch,units_str,b): # Sam's frankenstein - combines Kindsvater-Shen for when flows only in v-notch, then Bergmann 1963 combined equation when above rectanglular crest
#     # Q = C*ht**1.72 - d + e*b*hr # original Bergmann Eq - collapses to Cone Eq at H <=1 ft, just v-notch, hr=hm-ht
#     if hm <= 0:
#         Q_tot = 0
#     elif hm > 0 and hm <= (P_rect-P_vnotch):
#         Q_tot = Q_Kindsvater_Shen(angle_vnotch,hm,P_vnotch,B)
#     else: # Bergmann 1963 from USBR Water Msmt Manual 1997 and Quebec Ministry of the Environment 2007 - for combination v-notch/rectangular weirs - might be only for fully contracted v-notch weirs?
#         # Q_tot = Q_Bergmann(hm,P_rect,P_vnotch,units_str,b)
#         Q_vnotch = Q_Kindsvater_Shen(angle_vnotch,(P_rect-P_vnotch),P_vnotch,B)
#         ht = (P_rect-P_vnotch)
#         hr = hm-ht
#         if units_str == 'Imperial':
#             C = 3.9
#             d = 1.5
#             e = 3.3
#         elif units_str == 'SI':
#             C = 5.2
#             d = 1.5
#             e = 1.82
#         else:
#             print('Error: no units system specified.')
#         Q_tot = Q_vnotch + e*b*hr
#     return Q_tot

def Q_Holly_frankenstein(hm,P_rect,P_vnotch,B,angle_vnotch): # Holly's frankenstein - combines Kindsvater-Shen for when flows only in v-notch, FHWA's HY-8 stage-discharge curve for headwater of culvert minus HY-8 discharge at top of v-notch
    if hm <= 0:
        Q_tot = 0
    elif hm > 0 and hm <= (P_rect-P_vnotch):
        Q_tot = Q_Kindsvater_Shen(angle_vnotch,hm,P_vnotch,B)
    else: # uses stage-discharge curve made from HY-8, subtracts the flow at this depth in a 4' culvert from the flow at the depth of the top of v-notch in 4' Givens culvert
        Q_vnotch = Q_Kindsvater_Shen(angle_vnotch,(P_rect-P_vnotch),P_vnotch,B)
        Q_HY8 = 7.8791*(hm+4/12)**1.6098
        Q_HY8_topvnotch = 20.55 # cfs from HY8 at top of v-notch: 1.5ft plus 4in between v-notch crest and invert of culvert
        Q_top = Q_HY8 - Q_HY8_topvnotch
        Q_tot = Q_vnotch+Q_top
    return Q_tot

# vectorize the functions so they can be performed on whole arrays:
kh_vnotch_KC = np.vectorize(kh_vnotch_KC)
Q_Kindsvater_Shen = np.vectorize(Q_Kindsvater_Shen)
Ce_Phil = np.vectorize(Ce_Phil)
Q_weir_Phil_frankenstein = np.vectorize(Q_weir_Phil_frankenstein)
Q_Bergmann = np.vectorize(Q_Bergmann)
Q_Sam_frankenstein = np.vectorize(Q_Sam_frankenstein)
Q_Holly_frankenstein = np.vectorize(Q_Holly_frankenstein)

# if P < 2*H:
#     print('Warning: P < 2*H.')
# elif P < 0.2: # 0.2 ft:
#     print('Warning: P < 0.2 ft below crest.')
# elif H/P > 5: 
#     print('Warning: H/P > 5, weir is "no longer control section" (Sturm, 2001).')
# elif H > 2: # 2 ft
#     print('Warning: max head over partially-contracted v-notch of 2 ft exceeded.')
# else:
#     pass