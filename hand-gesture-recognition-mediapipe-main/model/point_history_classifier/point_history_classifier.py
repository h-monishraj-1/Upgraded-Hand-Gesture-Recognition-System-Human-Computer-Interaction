import numpy as np, tensorflow as tf

class PointHistoryClassifier:
    def __init__(self, model_path='model/point_history_classifier/point_history_classifier.tflite', score_th=0.5, invalid_value=0, num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(self, point_history):
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        result_index = np.argmax(np.squeeze(result))
        if np.squeeze(result)[result_index] < self.score_th:
            return self.invalid_value
        return result_index
