from sample import RBFKernel2
import numpy as np

data = np.array(
    [
        [0.5, 0.5, 1.0],
        [0.2, 0.2, 0.1],
        [1.0, 0.9, 0.8]
    ]
)


kernel = RBFKernel2(data)

print(kernel)
