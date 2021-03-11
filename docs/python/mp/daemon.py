# source: https://pymotw.com/3/multiprocessing/basics.html#daemon-processes
import multiprocessing
import time
import sys

def daemon():
    p = multiprocessing.current_process()
    print('Starting:', p.name, p.pid)
    sys.stdout.flush()
    time.sleep(2)
    print('Existing:', p.name, p.pid)
    # https://www.geeksforgeeks.org/python-sys-stdout-flush/
    sys.stdout.flush()

def non_daemon():
    p = multiprocessing.current_process()
    print('Starting:', p.name, p.pid)
    sys.stdout.flush()
    print('Existing:', p.name, p.pid)
    sys.stdout.flush()

if __name__ == '__main__':
    d = multiprocessing.Process(name = "daemon", target = daemon)
    d.daemon = True

    n = multiprocessing.Process(name = "non-daemon", target = non_daemon)
    n.daemon = False

    d.start()
    time.sleep(0.1)
    n.start()
# The output does not include the “Exiting” message from the daemon process,
# since all of the non-daemon processes (including the main program) exit
# before the daemon process wakes up from its 2 second sleep.
# The daemon process is terminated automatically before the main program
# exits, to avoid leaving orphaned processes running
#    time.sleep(3)
