import time
import numpy as np

def slow_function():
    for i in range(10):
        time.sleep(10)
        res = np.random.rand(3, 3)
        np.savetxt(f"res-{i}.txt", res)
    
slow_function()