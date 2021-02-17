import numpy
import pylab
import matplotlib
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = numpy.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
bins=pylab.hist(v, bins=50)       # matplotlib version (plot)
pylab.savefig('hist1.png',dpi=75)
pylab.show()
print(bins)
# Compute the histogram with numpy and then plot it
(n, bins) = numpy.histogram(v, bins=50)  # NumPy version (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.show()