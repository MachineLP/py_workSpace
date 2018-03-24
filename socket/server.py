import socket
import cv2
import numpy
from predict import HyqAlg, ProjectInterface
alg_core = HyqAlg(pb_path_1="model/frozen_model.pb")
result_dict = ProjectInterface({'test.jpg': 'test.jpg'}, proxy=alg_core)
print (result_dict)


address = ('127.0.0.1', 8002)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(True)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

conn, addr = s.accept()
while 1:
    length = recvall(conn,16)
    stringData = recvall(conn, int(length))
    data = numpy.fromstring(stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    cv2.imwrite('test.jpg', decimg)
    result_dict = ProjectInterface({'test.jpg': 'test.jpg'}, proxy=alg_core)
    print (result_dict)
    cv2.imshow('SERVER',decimg)
    if cv2.waitKey(10) == 27:
        break
    
s.close()
cv2.destroyAllWindows()