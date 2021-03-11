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

    # waits for the daemon to exit
    #d.join()
    d.join(1)
    print("d.is_alive() = ", d.is_alive())
