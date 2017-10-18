import numpy as np

a = np.array(range(10000))

b = [element for element in a if element < 5]
print(a)
mean = np.mean(a)
print(b)
print(mean)
print(enumerate([]))

a = [step for step, val in enumerate([])]
print(a)