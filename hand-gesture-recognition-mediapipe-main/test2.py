import csv, copy, argparse, itertools, sys
from collections import Counter, deque
import cv2 as cv, numpy as np, mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

class HandGestureRecognition:
    def __init__(self, args):
        self.args = args
        self.init_camera()
        self.init_mediapipe_hands()
        self.init_classifiers()
        self.init_csv()
        self.init_point_history()

    def init_camera(self):
        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height
        self.use_static_image_mode = self.args.use_static_image_mode

        self.input_source = self.get_input_source()
        if isinstance(self.input_source, cv.VideoCapture):
            self.cap = self.input_source
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        elif isinstance(self.input_source, str):
            self.image = cv.imread(self.input_source)
            if self.image is None:
                print(f"Error reading image from {self.input_source}. Exiting.")
                sys.exit()
            else:
                self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                self.cap_width, self.cap_height = self.image.shape[1], self.image.shape[0]

    def get_input_source(self):
        source_type = input("Choose input source (1 for webcam, 2 for image): ")
        if source_type == "1":
            return cv.VideoCapture(0)
        elif source_type == "2":
            image_path = input("Enter image path: ")
            return image_path
        else:
            print("Invalid input source. Exiting.")
            sys.exit()

    def init_mediapipe_hands(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.use_static_image_mode, max_num_hands=4,
                                          min_detection_confidence=self.args.min_detection_confidence,
                                          min_tracking_confidence=self.args.min_tracking_confidence)

    def init_classifiers(self):
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    def init_csv(self):
        self.csvFpsCalc = CvFpsCalc(buffer_len=10)

    def init_point_history(self):
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

    def main(self):
        while True:
            self.process_frame()

    def process_frame(self):
        fps = self.csvFpsCalc.get()
        key = cv.waitKey(10)

        if key == 27:
            sys.exit()
        number, mode = self.select_mode(key)

        if isinstance(self.input_source, cv.VideoCapture):
            ret, image = self.cap.read()
            if not ret:
                sys.exit()
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
        elif isinstance(self.input_source, str):
            debug_image = copy.deepcopy(self.image)

        results = self.hands.process(debug_image)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(debug_image, self.point_history)

                self.logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                debug_image = self.draw_bounding_rect(debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )

        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    def select_mode(self, key):
        number = -1
        mode = 0
        if 48 <= key <= 57:
            number = key - 48
        if key == 110:
            mode = 0
        if key == 107:
            mode = 1
        if key == 104:
            mode = 2
        return number, mode

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(map(abs, temp_landmark_list))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)

        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 255, 0), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 255, 0), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 255, 0), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 255, 0), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 255, 0), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 255, 0), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 255, 0), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            color = (0, 255, 0) if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19] else (255, 0, 0)
            size = 5 if index not in [4, 8, 12, 16, 20] else 8
            cv.circle(image, (landmark[0], landmark[1]), size, (0, 255, 0), -1)
            cv.circle(image, (landmark[0], landmark[1]), size, (255, 0, 0), 1)
        return image

    def draw_bounding_rect(self, image, brect):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text:
            info_text += ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if finger_gesture_text:
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (255, 255, 255), 2, cv.LINE_AA)
        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] and point[0]: cv.circle(image, (point[0], point[1]), 1 + index // 2, (152, 251, 152), 2)
        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                       1, cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
        return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960, help='cap width')
    parser.add_argument("--height", type=int, default=540, help='cap height')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7, help='min_detection_confidence')
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help='min_tracking_confidence')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    app = HandGestureRecognition(args)
    app.main()

