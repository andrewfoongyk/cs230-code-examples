import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

def var_fun(mu, sigma):
    r = mu/sigma
    return mu**2 * ((1 + 1/r**2) * norm.cdf(r) + 1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) \
         - (1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) + norm.cdf(r) )**2)

def var_over_mu2(r):
    return ((1 + 1/r**2) * norm.cdf(r) + 1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) \
         - (1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) + norm.cdf(r) )**2)

def mean_fun(mu, sigma):
    r = mu/sigma
    return mu * (1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) + norm.cdf(r))

def mean_over_mu(r):
    return (1/(r*np.sqrt(2 * np.pi)) * np.exp(-r**2/2) + norm.cdf(r))

# plot sd/mu and mean/mu against each other, for varying r
r = np.linspace(-3, 3, 1000)
x = mean_over_mu(r)
y = np.sqrt(var_over_mu2(r))
plt.scatter(x, y)

plt.scatter(x, y, c=r)
plt.colorbar()



x2 = np.linspace(-5, 5, 1000)
y2 = x2**2
#plt.plot(x2, y2, color='r')
plt.xlim(-3, 3)
plt.ylim(0, 2)
plt.xlabel('$\mathbb{E}[\phi(a)]/\mu$')
plt.ylabel('Std$[\phi (a)]/\mu$')
plt.savefig('mean_against_sd.pdf')

# num_samples = 10000000
# for mu in [-3,-2,-1,1e-10,1,2,3]:
#     for sigma in [1e-6,1e-3,1e-1,1,10,100]:
#         # simulate Gaussian rv's
#         a = np.random.normal(0, 1, size=(num_samples))
#         a = a * sigma + mu
#         # pass through ReLU
#         phi = np.maximum(0, a)
#         # calculate variance
#         var = np.var(phi)
#         # print
#         print('mu: {}'.format(mu))
#         print('sigma: {}'.format(sigma))
#         print('Estimated variance:{}'.format(var))
#         print('Calculated variance:{} \n'.format(var_fun(mu, sigma)))