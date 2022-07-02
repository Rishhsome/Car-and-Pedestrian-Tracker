import cv2

car_tracker = cv2.CascadeClassifier('cars.xml')

pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

video = cv2.VideoCapture('Car_Image.jpg')

while True:
    successful_frame_read, frame = video.read()

    if successful_frame_read:
        bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    car_coordinates = car_tracker.detectMultiScale(bw_img)

    pedestrian_coordinates = pedestrian_tracker.detectMultiScale(bw_img)

    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x+1, y+2),(x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y),(x+w, y+h), (71, 99, 255), 2)

    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Car & Pedestrian Detector App - Press E to EXIT', frame)

    key = cv2.waitKey(1)

    if key == 69 or key == 101:
        break

video.release()




