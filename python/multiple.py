import math
import threading as thread
import _thread
def is_prime(n):
    for i in range(2,int(math.sqrt(n))+1):
        if n%i ==0:
            return False
    return True
def find_prime(a,b):
    for n in range(a,b+1):
        if is_prime(n):
            print(n)
'''_thread.start_new_thread(find_prime,(100,300))
_thread.start_new_thread(find_prime,(301,500))
while 1:
    if thread.activeCount()==0:
        exit()'''
class myThread (thread.Thread):
    def __init__(self):
        thread.Thread.__init__(self)
    def run(self):
        find_prime(10,30000)
th1 = myThread()
th2 = myThread()
th1.start()
th2.start()
th1.join()
th2.join()
