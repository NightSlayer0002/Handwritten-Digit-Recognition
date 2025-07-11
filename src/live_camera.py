import os
import cv2
import tensorflow as tf
import numpy as np

def preprocess_frame(frame):   #prepare ROI to match the dataset
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[100:300, 100:300]
    roi = cv2.resize(roi, (28,28))
    roi = np.invert(roi)
    roi = roi/255.0
    return roi.reshape(1,28,28,1)

def live_detection():       #captures web frames and displays results
    model = tf.keras.models.load_model("models/handwritten.keras")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame)
        prediction = np.argmax(processed)
        #visualization
        cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
        cv2.putText(frame, f"Digit:{prediction}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Live MNIST Detection", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    print("press 'q' to quit")
    live_detection()