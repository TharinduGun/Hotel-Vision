import cv2

cap = cv2.VideoCapture("data/videos/video1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("test", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()