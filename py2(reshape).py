import numpy as np

mat = np.concatenate((np.zeros((3,1)), np.eye(3)), axis=1) + np.eye(3,4)*2

res = mat.reshape(12,1)
res1 = mat.flatten()
print(res)
