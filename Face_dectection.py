import cv2

cam = cv2.VideoCapture(0)
while True:
    
    success, frame = cam.read()  # read the camera frame
    # cv2.imsh1ow('WEBCAM', frame)
    print(success)
    if not success:
        break
    else:
        detector=cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
        faces=detector.detectMultiScale(frame,1.1,7)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow('', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break