import numpy as np
import numpy.random as r

def rzip(n=1, pi=0.5, lam=1):
    """
    Simulates n samples from a zero-inflated Poisson
    Returns 0 with pi probability
    Returns pois(lam) with 1 - pi probability
    """
    x = np.zeros(n)
    f = np.vectorize(lambda x: 0 if r.uniform() < pi else
                     r.poisson(lam))
    return f(x)


def rzin(n=1, pi=0.5, mu=0, sigma=1):
    """
    Simulates n samples from a zero-inflated positive Gaussian (normal)
    Returns 0 with pi probability
    Returns abs(normal(mu, sigma)) with 1 - pi probability
    """
    x = np.zeros(n)
    f = np.vectorize(lambda x: 0 if r.uniform() < pi else
                     abs(r.normal(mu, sigma)))
    return f(x)


