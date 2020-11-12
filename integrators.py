import matplotlib.pyplot as plt
import numpy as np
import math

def func(s, beta):  # define a derivative for the problem
    deriv = -beta * s * (100 - s) + 0.25 * (100 - s)
    return deriv

def diffy_q(I_k, k, p_k, lam, theta):
    # I_tilde_k = I_k_tilde(I_k, N_k)
    diff = lam*k*(p_k-I_k)*theta - I_k
    return diff

def I_k_tilde(I_k, N_k):
    return I_k/np.sum(N_k)

def euler(h, current_I_k, k, p_k, lam, theta):
    # current_I_tilde = current_I_k/np.sum(N_k)
    diffyq = diffy_q(current_I_k, k, p_k, lam, theta)
    newY = current_I_k + (h * diffyq)
    return newY

def get_theta(degree_dist, new_p_k, infected_vec):
    top_sum = 0
    bottom_sum = 0
    for k in range(len(new_p_k)):
        degree = int(math.floor(k/2))
        top_sum+=(degree*(infected_vec[k]))
        bottom_sum+=(degree*new_p_k[k])
    return top_sum/bottom_sum

def heun(h, maxT, init, beta, eulerY):
    heunY = [init]
    T = np.arange(0, maxT, h)
    for k in range(len(T) - 1):
        newY = heunY[k] + h * (func(heunY[k], beta) + func(eulerY[k + 1], beta))
        heunY.append(newY)
    return heunY
