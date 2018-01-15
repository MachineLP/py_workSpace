
# coding=utf-8

import threading
from time import ctime, sleep

# 多线程如何返回值
class MyThread(threading.Thread):

    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

# 多线程
def music(func):
	for i in range(2):
		print ("I was listening to %s. %s" %(func,ctime()))
		sleep(1)

def move(func):
	for i in range(2):
		print ("I was at the %s! %s" %(func,ctime()))
		sleep(5)

def add(a, b):
	#print ('a+b:', a+b)
	return a+b

threads = []
t1 = threading.Thread(target=music, args=(u'爱情买卖',))
threads.append(t1)
t2 = threading.Thread(target=move, args=(u'阿凡达',))
threads.append(t2)

t3 = MyThread(add, args=(1,2,))
threads.append(t3)

if __name__ == '__main__':
	for t in threads[:2]:
		t.setDaemon(True)
		t.start()

	for t in threads[2:]:
		t.setDaemon(True)
		t.start()
	print ("MyThread:a+b=%d ! %s" % (t.get_result(),ctime()))

	print ("all over %s" %ctime())
