from objectdetector import *


def main():
    video_path = 'Sample Video 2.mp4'
    configuration_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    model_path = 'frozen_inference_graph.pb'
    classes_path = 'coco.names'

    detector = ObjectDetector(video_path, configuration_path, model_path, classes_path)
    detector.capture_video()


if __name__ == '__main__':
    main()
