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

def compute_ll(X_train, mu, sigma, phi, n, J, K):
    W = compute_w(X_train, mu, sigma, phi, n, J, K)
    ll = np.zeros((len(X_train), 1))
    for i in range(len(X_train)):
        sumlog = 0
        for j in range(K):
            detJ = np.absolute(np.linalg.det(sigma[j]))
            numlog = (1 / (((2 * math.pi) ** (n / 2)) * math.sqrt(detJ))) * math.exp(
                (-0.5) * np.transpose(X_train[i] - mu[j]).dot(np.linalg.inv(sigma[j])).dot(X_train[i] - mu[j]) * phi[j])
            if W[i,j] != 0:
                sumlog = sumlog + W[i, j] * np.log(numlog/W[i, j])
        ll[i] = sumlog
    return ll


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
old_ll = 0
new_ll = 0
tol = 0.001
for iteration in range(10):
    old_ll = np.sum(compute_ll(X_train, mu, sigma, phi, n, J, K))
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
    new_ll = np.sum(compute_ll(X_train, mu, sigma, phi, n, J, K))

    if (np.abs(new_ll - old_ll) < tol):
        print("Convergence attained at iteration:",iteration+1)
        break

# print('phi', phi)
# print('mu', mu)
# print(mu)
# print(sigma)
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -compute_ll(XX, mu, sigma, phi, n, J, K)
Z = Z.reshape(X.shape)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
