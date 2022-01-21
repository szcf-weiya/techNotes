import time


def slow_function():
    for i in range(100):
        time.sleep(1)
        print(f"iter {i}: para = ...")


slow_function()