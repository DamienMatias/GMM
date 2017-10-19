import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.colors import LogNorm

def compute_w(X_train, mu, sigma, phi, n, J, K):
    W = np.zeros((len(X_train), J))
    for i in range(len(X_train)):
        for j in range(J):
            denom = 0
            detJ = np.absolute(np.linalg.det(sigma[j]))
            num = (1 / (((2 * math.pi) ** (n/2)) * math.sqrt(detJ))) * math.exp(
                (-0.5) * np.transpose(X_train[i] - mu[j]).dot(np.linalg.inv(sigma[j])).dot(X_train[i] - mu[j]) * phi[j])
            for l in range(K):
                detl = np.absolute(np.linalg.det(sigma[l]))
                denom = denom + (1 / (((2 * math.pi)**(n/2)) * math.sqrt(detl))) * math.exp(
                    (-0.5) * np.transpose(X_train[i] - mu[l]).dot(np.linalg.inv(sigma[l])).dot(X_train[i] - mu[l]) * phi[l])
            W[i, j] = num / denom
    return W


m_samples = 300
n = 2
J = 2
K = 2
# generate random sample, two components
np.random.seed(0)
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(m_samples, n) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(m_samples, n), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
# print(X_train)

mu = np.array([random.choice(X_train), random.choice(X_train)])
sigma = np.array([np.random.randn(2, 2), np.random.randn(2, 2)])
phi = [0.4, 0.6]

for iteration in range(10):
    # E-step
    W = compute_w(X_train, mu, sigma, phi, n, J, K)

    # M-step
    for j in range(J):
        phi[j] = (1/(len(X_train))) * np.sum(W[:, j])

    for j in range(J):
        numu = 0
        denum = 0
        for i in range(len(X_train)):
            numu = numu + (W[i, j] * (X_train[i]))
            denum = denum + W[i, j]
        mu[j] = numu/denum

    for j in range(J):
        numsigma = 0
        densigma = 0
        for i in range(len(X_train)):
            numsigma = numsigma + W[i, j] * np.outer(X_train[i] - mu[j], X_train[i] - mu[j])
            densigma = densigma + W[i, j]
        sigma[j] = numsigma / densigma

    # Likelihood

# print('phi', phi)
# print('mu', mu)
print(mu)
print(sigma)
