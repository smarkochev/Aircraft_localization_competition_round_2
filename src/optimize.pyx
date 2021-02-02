# load libraries
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs, exp, sqrt, cos, sin, pi, pow

from cython import boundscheck, wraparound
from cython.parallel import prange


# WSG84 ellipsoid constants
cdef:
    double wgs_a = 6378137  # [m]
    double wgs_e2 = 0.00669437999014

cdef:
    double light_speed = 299792458.


# --------------------------------------------------------------------
# Cython function to convert WGS84 coordinates to cartesian
# --------------------------------------------------------------------

cdef wgs2cart_c(double latitude, double longitude, double height):
    cdef:
        double lat = pi * latitude / 180.
        double lon = pi * longitude / 180.
        
        # prime vertical radius of curvature
        double N = wgs_a / sqrt(1 - wgs_e2 * pow(sin(lat), 2))
        double cart[3]

    cart[0] = (N + height) * cos(lat) * cos(lon)
    cart[1] = (N + height) * cos(lat) * sin(lon)
    cart[2] = ((1 - wgs_e2) * N + height) * sin(lat)
    
    return cart
    

    # --------------------------------------------------------------------
# Cython function to calculate distance in 3D space
# --------------------------------------------------------------------

cdef dist3d_c(double latitude1, double longitude1, double height1,
              double latitude2, double longitude2, double height2):
    
    cdef double cart1[3], cart2[3]
    
    cart1 = wgs2cart_c(latitude1, longitude1, height1)
    cart2 = wgs2cart_c(latitude2, longitude2, height2)
    
    return sqrt(pow(cart1[0] - cart2[0], 2) + pow(cart1[1] - cart2[1], 2) + pow(cart1[2] - cart2[2], 2))



cdef inline double eff_velocity(double h_min, double h_max, double A0, double B) nogil:
    if h_min < h_max:
        return light_speed / (1 + A0 * (exp(-B*h_min) - exp(-B*h_max)) / B / (h_max - h_min))
    elif h_min > h_max:
        return light_speed / (1 + A0 * (exp(-B*h_max) - exp(-B*h_min)) / B / (h_min - h_max))
    else:
        return light_speed / (1 + A0 / B)
    
    
    
cdef inline double eff_velocity_test(double A0) nogil:
    return light_speed / (1 + A0)
    

    
    
@boundscheck(False)
@wraparound(False)
def optimize_shifts(double[:] x,
                       double[:] points,
                       double[:] heights,
                       double[:] st_loc,
                       double[:] st_hgt,
                       int[:] st_1,
                       int[:] st_2,
                       double[:] t_1,
                       double[:] t_2,
                       double A0,
                       double B,
                       int loss_l1=1,
                       int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        
        double[:] st_shift = np.array([10**y for y in x])
        
    for i in prange(N, nogil=True, schedule='guided', num_threads=4):
        j1 = st_1[i]
        j2 = st_2[i]
        
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)
    
    
    
@boundscheck(False)
@wraparound(False)
def optimize_AB_shifts(double[:] x,
                       double[:] points,
                       double[:] heights,
                       double[:] st_loc,
                       double[:] st_hgt,
                       int[:] st_1,
                       int[:] st_2,
                       double[:] t_1,
                       double[:] t_2,
                       int loss_l1=1,
                       int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        
        double A0 = 10**x[0]
        double B = 10**x[1]
        double[:] st_shift = np.array([10**y for y in x[2:]])
        
    for i in prange(N, nogil=True, schedule='guided', num_threads=4):
        j1 = st_1[i]
        j2 = st_2[i]
        
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)
    

    
@boundscheck(False)
@wraparound(False)
def optimize_locations(double[:] x,
                       double[:] points,
                       double[:] heights,
                       int[:] st_1,
                       int[:] st_2,
                       double[:] t_1,
                       double[:] t_2,
                       int loss_l1=1,
                       int return_errors=0):
    
    cdef:
        int N = t_1.shape[0]
        int M = int((x.shape[0] - 2) / 4)
        int i, j1, j2
        double d1, d2, dt
        
        double[:] err = np.zeros(N)
        
        double A0 = 10**x[0]
        double B = 10**x[1]
        double[:] st_shift = np.array([pow(10, y) for y in x[2:M+2]])
        double[:] st_loc = np.zeros(3*M)
        double[:] st_hgt = np.zeros(M)
        
    for i in range(M):
        j1 = 3*i+M+2
        cart = wgs2cart_c(x[j1], x[j1+1], x[j1+2])
        st_loc[3*i] = cart[0]
        st_loc[3*i+1] = cart[1]
        st_loc[3*i+2] = cart[2]
        st_hgt[i] = x[j1+2]
    
    for i in prange(N, nogil=True, schedule='guided', num_threads=4):
        j1 = st_1[i]
        j2 = st_2[i]
        
        d1 = sqrt((points[3*i]-st_loc[3*j1])**2 + (points[3*i+1]-st_loc[3*j1+1])**2 + (points[3*i+2]-st_loc[3*j1+2])**2)
        d2 = sqrt((points[3*i]-st_loc[3*j2])**2 + (points[3*i+1]-st_loc[3*j2+1])**2 + (points[3*i+2]-st_loc[3*j2+2])**2)
        dt = eff_velocity(0.5*st_hgt[j1]+0.5*st_hgt[j2], heights[i], A0, B) * (t_1[i] - t_2[i] + st_shift[j1] - st_shift[j2])
        
        if loss_l1 == 1 or return_errors == 1:
            err[i] = fabs(d1 - d2 - dt)
        else:
            err[i] = (d1 - d2 - dt)**2
        
    if return_errors == 1:
        return np.array(err)
    elif loss_l1 == 1:
        return np.sum(err) / N
    else:
        return np.sqrt(np.sum(err) / N)

    
@boundscheck(False)
@wraparound(False)
def solve_point(double[:] x,
                double baroAlt,
                double[:] times,
                int[:] st,
                double[:] st_cart,
                double[:] st_hgt,
                double A0,
                double B,
                int loss_l1 = 1
                ):
    
    cdef:
        int N = times.shape[0]
        
        double err = 0
        
        int i, j, j1, j2
        double dist1, dist2, t1, t2
        double cart[3]
    
    cart = wgs2cart_c(x[0], x[1], baroAlt)
    
    for i in prange(N, nogil=True, schedule='guided', num_threads=4):
        j1 = st[i]
        dist1 = sqrt((st_cart[3*j1] - cart[0])**2 + (st_cart[3*j1+1] - cart[1])**2 + (st_cart[3*j1+2] - cart[2])**2)
        t1 = times[i] - dist1 / eff_velocity(st_hgt[j1], baroAlt, A0, B)
        
        
        for j in range(i+1, N):
            j2 = st[j]
            
            dist2 = sqrt((st_cart[3*j2] - cart[0])**2 + (st_cart[3*j2+1] - cart[1])**2 + (st_cart[3*j2+2] - cart[2])**2)
            t2 = times[j] - dist2 / eff_velocity(st_hgt[j2], baroAlt, A0, B)
            
            err += fabs(t1 - t2)
    
    return err / N
