from flask import Flask, render_template, Response, jsonify, request
import os
import gc
import threading
import time
from functools import lru_cache

# Set OpenCV environment variables before importing cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import json
import math
import traceback

# Configure TensorFlow for low memory usage
try:
    tf.config.experimental.enable_memory_growth = True
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except Exception as e:
    print(f"TensorFlow GPU configuration failed (using CPU): {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the sign language app instance
sl_app = None
app_lock = threading.Lock()

class SignLanguageApp:
    def __init__(self):
        try:
            # Load model with memory optimization
            self.model = load_model('model/cnn8grps_rad1_model.h5', compile=False)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Initialize MediaPipe with minimal resources
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Simple word suggestions instead of enchant (memory heavy)
        self.common_words = {
            'a': ['and', 'are', 'all', 'any', 'ask'],
            'b': ['be', 'but', 'by', 'big', 'boy'],
            'c': ['can', 'come', 'call', 'car', 'cat'],
            'd': ['do', 'day', 'did', 'dog', 'down'],
            'e': ['eat', 'end', 'eye', 'ear', 'egg'],
            'f': ['for', 'from', 'fun', 'far', 'fly'],
            'g': ['go', 'get', 'good', 'girl', 'give'],
            'h': ['he', 'his', 'her', 'how', 'home'],
            'i': ['is', 'it', 'in', 'if', 'into'],
            'j': ['job', 'joy', 'jump', 'just', 'join'],
            'k': ['know', 'keep', 'kind', 'key', 'kiss'],
            'l': ['like', 'look', 'love', 'let', 'long'],
            'm': ['me', 'my', 'make', 'man', 'more'],
            'n': ['no', 'not', 'now', 'new', 'name'],
            'o': ['of', 'on', 'or', 'out', 'old'],
            'p': ['put', 'play', 'please', 'pretty', 'person'],
            'q': ['question', 'quick', 'quiet', 'quite', 'queen'],
            'r': ['run', 'red', 'right', 'read', 'room'],
            's': ['see', 'she', 'say', 'some', 'stop'],
            't': ['the', 'to', 'that', 'this', 'time'],
            'u': ['up', 'us', 'use', 'under', 'until'],
            'v': ['very', 'visit', 'voice', 'view', 'video'],
            'w': ['we', 'with', 'will', 'what', 'where'],
            'x': ['x-ray', 'xbox', 'xerox', 'xylophone', 'extra'],
            'y': ['you', 'yes', 'your', 'year', 'young'],
            'z': ['zoo', 'zero', 'zone', 'zip', 'zoom']
        }
        
        self.offset = 29
        # Initialize all variables
        self.str = " "
        self.word = " "
        self.current_symbol = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.ten_prev_char = [" "] * 10
        self.count = -1
        self.prev_char = ""
        self.ccc = 0
        self.next_count = 0  # Counter for consecutive "next" gestures
        self.pts = []
        
        # Create white background image
        self.white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.imwrite("white.jpg", self.white)

    def get_word_suggestions(self, partial_word):
        """Get word suggestions based on partial input using lightweight dictionary"""
        suggestions = []
        if len(partial_word) > 0:
            first_char = partial_word[0].lower()
            if first_char in self.common_words:
                # Find words that start with the partial word
                for word in self.common_words[first_char]:
                    if word.startswith(partial_word):
                        suggestions.append(word)
                
                # If we don't have enough exact matches, add other words starting with same letter
                if len(suggestions) < 4:
                    for word in self.common_words[first_char]:
                        if word not in suggestions:
                            suggestions.append(word)
                            if len(suggestions) >= 4:
                                break
        
        return suggestions[:4]  # Return max 4 suggestions

    def clear_word(self):
        self.str = " "
        self.word = " "
        self.current_symbol = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.ten_prev_char = [" "] * 10
        self.count = -1
        self.prev_char = ""
        self.ccc = 0
        self.next_count = 0  # Reset consecutive "next" counter

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def predict_gesture(self, frame):
        try:
            if self.model is None:
                return None, cv2.imread("white.jpg")
                
            frame = cv2.flip(frame, 1)
            # Resize frame to reduce memory usage
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            bonestructure_img = cv2.imread("white.jpg")
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    lmList.append([int(lm.x * w), int(lm.y * h)])
                self.pts = lmList
                
                # Bounding box
                x_list = [pt[0] for pt in lmList]
                y_list = [pt[1] for pt in lmList]
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                x, y, w_box, h_box = x_min, y_min, x_max - x_min, y_max - y_min
                
                # Crop hand region
                x1 = max(x - self.offset, 0)
                y1 = max(y - self.offset, 0)
                x2 = min(x + w_box + self.offset, frame.shape[1])
                y2 = min(y + h_box + self.offset, frame.shape[0])
                hand_w, hand_h = max(x2 - x1, 1), max(y2 - y1, 1)
                os = max(((400 - hand_w) // 2) - 15, 0)
                os1 = max(((400 - hand_h) // 2) - 15, 0)
                
                # Draw connections and points on white background, clamp coordinates
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    x1s = int(np.clip(lmList[start_idx][0] - x1 + os, 0, 399))
                    y1s = int(np.clip(lmList[start_idx][1] - y1 + os1, 0, 399))
                    x2s = int(np.clip(lmList[end_idx][0] - x1 + os, 0, 399))
                    y2s = int(np.clip(lmList[end_idx][1] - y1 + os1, 0, 399))
                    cv2.line(bonestructure_img, (x1s, y1s), (x2s, y2s), (0, 255, 0), 3)
                
                for i in range(21):
                    px = int(np.clip(lmList[i][0] - x1 + os, 0, 399))
                    py = int(np.clip(lmList[i][1] - y1 + os1, 0, 399))
                    cv2.circle(bonestructure_img, (px, py), 2, (0, 0, 255), 1)
                
                self.pts = [[int(np.clip(lmList[i][0] - x1 + os, 0, 399)), int(np.clip(lmList[i][1] - y1 + os1, 0, 399))] for i in range(21)]
                self.predict(bonestructure_img)
                
                # Clean up intermediate variables
                del rgb, results, hand_landmarks, lmList, x_list, y_list
                
                return self.current_symbol, bonestructure_img
            
            # If no hand detected, return blank white
            del rgb, results
            return None, bonestructure_img
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            traceback.print_exc()
            return None, cv2.imread("white.jpg")

    def predict(self, test_image):
        # Use the same logic as before, but skip if not enough points
        if len(self.pts) != 21:
            self.current_symbol = "Empty"
            return "Empty"
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # All the conditions from final_pred.py
        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]:
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + 13 < self.pts[8][0] and self.pts[0][0] + 13 < self.pts[12][0] and self.pts[0][0] + 13 < self.pts[16][0] and
                    self.pts[0][0] + 13 < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for subgroups  starts
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        # Print for debugging
        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        # Backspace logic
        if ch1 in ['next', 'B', 'C', 'H', 'F', 'X']:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        # Handle next and backspace
        if ch1 == "next" and self.prev_char != "next":
            self.next_count += 1
            if self.next_count >= 2:
                # Add space after showing "next" twice, but keep the current word
                self.str = self.str + " "
                self.next_count = 0  # Reset counter after adding space
                # Don't clear the word, just update it for display
                if len(self.str.strip()) != 0:
                    st = self.str.rfind(" ")
                    ed = len(self.str)
                    word = self.str[st+1:ed]
                    self.word = word
            else:
                if self.ten_prev_char[(self.count-2)%10] != "next":
                    if self.ten_prev_char[(self.count-2)%10] == "Backspace":
                        self.str = self.str[0:-1]
                    else:
                        if self.ten_prev_char[(self.count-2)%10] != "Backspace":
                            self.str = self.str + self.ten_prev_char[(self.count-2)%10]
                else:
                    if self.ten_prev_char[(self.count-0)%10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-0)%10]
        elif ch1 != "next":
            # Reset next counter if gesture is not "next"
            self.next_count = 0

        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count%10] = ch1

        # Word suggestion logic using lightweight dictionary
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st+1:ed]
            self.word = word
            if len(word.strip()) != 0:
                suggestions = self.get_word_suggestions(word.lower())
                self.word1 = suggestions[0] if len(suggestions) > 0 else " "
                self.word2 = suggestions[1] if len(suggestions) > 1 else " "
                self.word3 = suggestions[2] if len(suggestions) > 2 else " "
                self.word4 = suggestions[3] if len(suggestions) > 3 else " "
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

        return ch1

# --- Flask routes and app runner ---

def initialize_app():
    """Initialize the sign language app instance with memory management"""
    global sl_app
    with app_lock:
        if sl_app is None:
            try:
                sl_app = SignLanguageApp()
                print("Sign Language App initialized successfully")
            except Exception as e:
                print(f"Error initializing app: {e}")
                raise e

@app.route('/')
def index():
    try:
        initialize_app()
        return render_template('index.html',
            current_symbol=sl_app.current_symbol,
            word=sl_app.word,
            word1=sl_app.word1,
            word2=sl_app.word2,
            word3=sl_app.word3,
            word4=sl_app.word4,
            str=sl_app.str)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html',
            current_symbol="Error",
            word="",
            word1="",
            word2="",
            word3="",
            word4="",
            str="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        initialize_app()
        if sl_app.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        img_data = data['image']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
            
        symbol, _ = sl_app.predict_gesture(frame)
        
        # Clean up memory
        del frame, nparr, img_bytes
        gc.collect()
        
        return jsonify({
            'current_symbol': sl_app.current_symbol,
            'word': sl_app.word,
            'word1': sl_app.word1,
            'word2': sl_app.word2,
            'word3': sl_app.word3,
            'word4': sl_app.word4,
            'str': sl_app.str
        })
    except Exception as e:
        print(f"Error in predict route: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        initialize_app()
        if sl_app.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        img_data = data['image']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
            
        symbol, processed_img = sl_app.predict_gesture(frame)
        
        img_base64 = None
        if processed_img is not None:
            _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            del buffer
        
        # Clean up memory
        del frame, nparr, img_bytes, processed_img
        gc.collect()
        
        return jsonify({
            'current_symbol': sl_app.current_symbol,
            'word': sl_app.word,
            'word1': sl_app.word1,
            'word2': sl_app.word2,
            'word3': sl_app.word3,
            'word4': sl_app.word4,
            'str': sl_app.str,
            'processed_image': img_base64
        })
    except Exception as e:
        print(f"Error in process_frame route: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Frame processing failed'}), 500

@app.route('/clear_word', methods=['POST'])
def clear_word():
    try:
        initialize_app()
        sl_app.clear_word()
        gc.collect()  # Clean up memory
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in clear_word route: {e}")
        return jsonify({'error': 'Clear word failed'}), 500

@app.route('/add_to_sentence', methods=['POST'])
def add_to_sentence():
    try:
        data = request.get_json()
        if not data or 'word' not in data:
            return jsonify({'error': 'No word provided'}), 400
            
        word = data.get('word', '')
        return jsonify({'status': 'success', 'word_added': word})
    except Exception as e:
        print(f"Error in add_to_sentence route: {e}")
        return jsonify({'error': 'Add to sentence failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'service': 'sign-language-recognition'})

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    # Optional: Start memory monitoring if requested
    monitor_memory = os.environ.get("MONITOR_MEMORY", "false").lower() == "true"
    
    if monitor_memory:
        try:
            # Only try to import memory monitoring in development
            import importlib.util
            spec = importlib.util.find_spec("memory")
            if spec is not None:
                from memory import MemoryMonitor
                print("🔍 Starting integrated memory monitoring...")
                monitor = MemoryMonitor()
                monitor.start_monitoring(interval=10)
                print("📊 Memory monitoring started (10s intervals)")
            else:
                print("⚠️ Memory monitoring module not found (development only)")
        except ImportError:
            print("⚠️ Memory monitoring not available (install psutil)")
        except Exception as e:
            print(f"⚠️ Could not start memory monitoring: {e}")
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🌐 Access your app at: http://localhost:{port}")
    if monitor_memory:
        print("📋 Memory usage will be logged to: memory_usage.log")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)