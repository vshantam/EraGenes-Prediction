import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import urllib.request
import keras.models
import pickle
import math
import imutils
from imutils import face_utils
import dlib


class keyandlandmark(object):

    @classmethod
    def rect_to_bb(self, rect):
        x = rect.left()

        y = rect.top()

        w = rect.right() - x

        h = rect.bottom() - y

        return (x, y, w, h)

    @classmethod
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    @classmethod
    def get_detector(self, path):
        detector = dlib.get_frontal_face_detector()

        predictor = dlib.shape_predictor(path)

        return detector, predictor


class FaceCV(object):

    CASE_PATH = "pretrained_models/haarcascade_frontalface_default.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, thickness=1):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    @classmethod
    def crop_face(self, imgarray, section, margin=40, size=64):

        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        #url = "http://192.168.43.1:8080/shot.jpg"
        # infinite loop, break by key ESC
        KNOWN_DISTANCE = 14

        KNOWN_WIDTH = 5

        IMAGE_PATHS = []
        image = cv2.imread("")#use your own training pic

        #marker = find_marker(image)

        #focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            #imgResp = urllib.request.urlopen(url)
            # reading the frame
            obj = keyandlandmark()

            path = 'shape_predictor_68_face_landmarks.dat'

            detector, predictor = obj.get_detector(path)

            ret, frame = video_capture.read()
            #imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

            #frame = cv2.imdecode(imgNp,-1)
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray1, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=9,
                minSize=(self.face_size, self.face_size)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=0, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                face_imgs[i, :, :, :] = face_img

                for (i, rect) in enumerate(rects):

                    shape = predictor(gray1, rect)

                    shape = face_utils.shape_to_np(shape)

                    (x1, y1, w1, h1) = face_utils.rect_to_bb(rect)

                    for (x1, y1) in shape:
                        cv2.circle(frame, (x1, y1), 1, (0, 255, 255), -2)

            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, face in enumerate(faces):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "Female" if predicted_genders[i][0] > 0.5 else "Male")
                self.draw_label(frame, (x, y - 20), label)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args




def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()


if __name__ == "__main__":
    print(__doc__)
    main()
