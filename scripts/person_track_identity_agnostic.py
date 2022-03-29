import numpy as np
import cv2
import torch
import pickle 
from sort import Sort
import time
import detect_face 
import argparse
from attrdict import AttrDict
import model_utils

def load_pkl(path):
    with open(path, 'rb') as open_file:
        return pickle.load(open_file)

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', '-c', type=str)
    parser.add_argument('--num_persons_to_track', '-np', type=int)

    args = parser.parse_args()
    cfg = AttrDict(model_utils.load_cfg(args.cfg_name))

    return args, cfg


def main():
    args, cfg = parse_commandline()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #init detector
    model = detect_face.load_model(cfg.model_path, device)

    tracker =  Sort(use_dlib=False) 

    np.set_printoptions(suppress=True)

    # open camera and start detection
    cap = cv2.VideoCapture(0)
    frame_num = 0

    while True:
        begin_time = time.time()
        _, frame = cap.read()

        # image downsampled to size (320, 240)
        dim, scaling_fac = model_utils.resize_img(frame.shape)
        
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        detections = detect_face.detect_one(model, frame, device)

        # If no face is detected don't do anything
        if len(detections) == 0:
            print("No face detected sorry")
            pass
        # Waiting for camera to warm up
        if frame_num < 10:
            print("waiting for camera to get ready")
            frame_num += 1
            continue
        
         # Tracking the detections
        trackers = tracker.update(detections, embeddings=None)
        
        # Check fps
        fps = model_utils.check_fps(begin_time)


        # Draw rectanges for all the detected faces    
        for d in trackers:
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (2, 255, 0), 1)

        left_most_ids, centroids = model_utils.leftmost_n(np.array(trackers), args.num_persons_to_track)
        if left_most_ids is None:
            pass
        
        tracking_point = model_utils.find_mean_point(centroids)
        
        if tracking_point is None:
            pass
        else:
            cv2.circle(frame, (int(tracking_point[0]),int(tracking_point[1])), radius=0, color=(0, 0, 255), thickness=3)
        
            print((int(tracking_point[0]),int(tracking_point[1])))
        
        frame_num = frame_num + 1
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recordings
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    cap.release()


if __name__ == '__main__':
    main()
