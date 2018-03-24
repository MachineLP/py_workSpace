import socket
import cv2
import numpy

address = ('127.0.0.1', 8002)
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect(address)

capture = cv2.VideoCapture(0)
ret, frame = capture.read()
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

while ret:
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    sock.send( str(len(stringData)).ljust(16));
    sock.send( stringData );
    ret, frame = capture.read()
    #decimg=cv2.imdecode(data,1)
    #cv2.imshow('CLIENT',decimg)
    if cv2.waitKey(10) == 27:
        break
sock.close()
cv2.destroyAllWindows()

