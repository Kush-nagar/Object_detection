from detector import *
import os

def main():
    videoPath = '0'  # Change this to the path of your image or video file, or use '0' for webcam

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)

    if videoPath == '0':
        detector.videoPath = 0
        detector.onVideo()
    elif videoPath.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.onVideo()
    else:
        detector.onImage()

if __name__ == '__main__':
    main()
