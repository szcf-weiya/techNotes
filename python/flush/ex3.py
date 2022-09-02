import sys
import time

for i in range(3):
    print(i, end=' ')
    sys.stdout.flush()

    ## equivalent to
    #print(i, end=' ', flush=True)

    time.sleep(i)