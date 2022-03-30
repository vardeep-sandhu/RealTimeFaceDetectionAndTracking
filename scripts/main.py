import numpy as np
import cv2
import torch
import pickle 
from sort import Sort
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import sys
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
    parser.add_argument('--cfg_name', '-c', type=str, default="identity_tracker.yaml")
    parser.add_argument('-name', type=str)

    args = parser.parse_args()
    cfg = AttrDict(model_utils.load_cfg(args.cfg_name))

    return args, cfg



def main():
    args, cfg = parse_commandline()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detect_face.load_model(cfg.model_path, device)

    # Classes being tracked atm
    int2labels = model_utils.load_classes_dict(cfg.classes_list_path)
    print("Classes detected in this tracking sequence")
    print(int2labels)
    
    #init embedder and tracker 
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    tracker =  Sort(use_dlib=False) #create instance of the SORT tracker
    
    print ("Kalman tracker activated!")
    np.set_printoptions(suppress=True)

    # open camera and start detection
    cap = cv2.VideoCapture(0)
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((160, 160)),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
    frame_num = 0
    print(f"Tracking {args.name}")

    while True:
        begin_time = time.time()
        _, frame = cap.read()

        dim, scaling_fac = model_utils.resize_img(frame.shape)
        
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        detections = detect_face.detect_one(model, frame, device)

        if frame_num < 10:
            print("waiting for camera to get ready")
            frame_num += 1
            continue
        
        # If no face is detected don't do anything
        if len(detections) == 0:
            print("No face detected sorry")
            pass
        else:
            cropped_img_tensor = model_utils.get_cropped_faces(detections, frame, transform)

            with torch.no_grad():
                vec = embedder(cropped_img_tensor.to(device))
                embeddings = vec.to("cpu")
        
            # Tracking the embeddings
            trackers = tracker.update(detections, embeddings=embeddings)
        # Check fps
        fps = model_utils.check_fps(begin_time)
        
        # Draw things on the image
        cv2.line(frame, (dim[0]//2, 0), (dim[0]//2, dim[1]), color=(0, 0, 0), thickness=3)
        cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        for d in trackers:
            
            if int2labels[int(d[4])] == args.name:    
                cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (2, 255, 0), 1)
                cv2.putText(frame, args.name, (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                bb_x = int((((d[0] + d[2]) / 2) / dim[0]) * 640)
            else:
                if (frame_num % 5) == 0:
                    print("Frame num", frame_num)
                    print(int2labels[int(d[4])])
            
        frame_num = frame_num + 1
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recordings
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    cap.release()


if __name__ == '__main__':
    main()
