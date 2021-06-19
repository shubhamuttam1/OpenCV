import cv2
import mediapipe as mp
import time


class hand_detector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.max_hands, self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, hand_number=0, draw=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = frame.shape  # c = channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

        return lm_list


def main():
    previous_time = 0
    current_time = 0

    cap = cv2.VideoCapture(0)
    detector = hand_detector()

    while True:
        success, frame = cap.read()
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        if len(lm_list) != 0:
            print(lm_list[4])
            cv2.circle(frame, (lm_list[4][1], lm_list[4][2]),
                   radius=5, color=(0, 0, 255), cv2.FILLED)
# image = cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=-1)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(frame, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow('WEBCAM', frame)
        # cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()