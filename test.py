
import numpy as np

a = [1,2,3,4]
b = [1,2,3,4,5]
c = [1, 2, 3, 4, 5, 6]
l = []
l.append(np.array(a))
l.append(np.array(b))
l.append(np.array(c))
l = np.concatenate(l)
print(l)
print(np.mean(l))


