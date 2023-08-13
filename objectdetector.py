import cv2
import numpy as np
import time

np.random.seed(20)


class ObjectDetector:
    def __init__(self, video_path, configuration_path, model_path, classes_path):
        self.video_path = video_path
        self.configuration_path = configuration_path
        self.model_path = model_path
        self.classes_path = classes_path

        # Configure the object detection network using OpenCV's dnn module
        self.network = cv2.dnn_DetectionModel(self.model_path, self.configuration_path)
        self.network.setInputSize(320, 320)  # Set input size for model
        self.network.setInputScale(1.0 / 127.5)  # Normalize input scale
        self.network.setInputMean((127.5, 127.5, 127.5))  # Set input mean values
        self.network.setInputSwapRB(True)  # Swap color channels

        # Load class names and assign random colors to each one for visualization
        self.load_classes()

    def load_classes(self):
        with open(self.classes_path, 'r') as file:
            self.classes_list = file.read().splitlines()

        self.classes_list.insert(0, '__Background__')
        self.colors_list = np.random.uniform(low=0, high=256, size=(len(self.classes_list), 3))  # Assign random colors

    def capture_video(self):
        # Open the video file for capturing frames
        video_capture = cv2.VideoCapture(self.video_path)

        if not video_capture.isOpened():
            print('Error opening the video. Please try again.')
            return

        (success, image) = video_capture.read()
        start_time = 0

        while success:
            # Calculate frame rate
            current_time = time.time()
            fps = 1 / (current_time - start_time)
            start_time = current_time

            # Perform object detection on the current frame
            class_ids, class_confidences, bounding_boxes = self.network.detect(image, confThreshold=0.5)
            bounding_boxes = list(bounding_boxes)
            class_confidences = list(np.array(class_confidences).reshape(1, -1)[0])
            class_confidences = list(map(float, class_confidences))
            bounding_boxes_indices = cv2.dnn.NMSBoxes(bounding_boxes, class_confidences, score_threshold=0.5,
                                                      nms_threshold=0.2)

            if len(bounding_boxes_indices) != 0:
                for i in range(0, len(bounding_boxes_indices)):
                    # Retrieve detection details for the current bounding box
                    bounding_box = bounding_boxes[np.squeeze(bounding_boxes_indices[i])]
                    class_confidence = class_confidences[np.squeeze(bounding_boxes_indices[i])]
                    class_id = int(class_ids[np.squeeze(bounding_boxes_indices[i])])
                    class_label = self.classes_list[class_id]
                    class_color = [int(c) for c in self.colors_list[class_id]]

                    # Format displayed text
                    displayed_text = '{}:{:.2f}'.format(class_label, class_confidence)

                    # Draw bounding boxes, labels, and lines around detected objects
                    x, y, w, h = bounding_box
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=class_color, thickness=2)
                    cv2.putText(image, displayed_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, class_color, 2)

                    # Calculate line width based on bounding box dimensions
                    line_width = min(int(w * 0.3), int(h * 0.3))

                    # Draw lines around the detected object for better visualization
                    cv2.line(image, (x, y), (x + line_width, y), class_color, thickness=5)
                    cv2.line(image, (x, y), (x, y + line_width), class_color, thickness=5)
                    cv2.line(image, (x + w, y), (x + w - line_width, y), class_color, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y + line_width), class_color, thickness=5)
                    cv2.line(image, (x, y + h), (x + line_width, y + h), class_color, thickness=5)
                    cv2.line(image, (x, y + h), (x, y + h - line_width), class_color, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w - line_width, y + h), class_color, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - line_width), class_color, thickness=5)

            # Display FPS and image
            cv2.putText(image, 'FPS: ' + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Object Detection System', image)

            # Check for user input to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = video_capture.read()

        video_capture.release()
        cv2.destroyAllWindows()
