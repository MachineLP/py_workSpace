
# coding:utf-8

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output4.mp4',fourcc, 20.0, (400, 400))

for weight in weights_history:
    img = getCAM(weight)
    out.write(img)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
