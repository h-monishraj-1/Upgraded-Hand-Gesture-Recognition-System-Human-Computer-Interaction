import numpy as np, tensorflow as tf

class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(np.squeeze(result))
