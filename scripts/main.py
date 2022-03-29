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

def load_pkl(path):
    with open(path, 'rb') as open_file:
        return pickle.load(open_file)


def main():

    if len(sys.argv) != 2:
        print("Please add name to track from [Vardeep, Leticia, Migel]")
        sys.exit()
    person_track = sys.argv[1]

    model_path = "weights/yolov5n-0.5.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #init detector
    model = detect_face.load_model(model_path, device)

    # Classes being tracked atm
    classes_list = np.loadtxt("classes.txt", dtype=str)
    int2labels = {int(label):cls for label, cls in classes_list}
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
    print(f"Tracking {person_track}")

    while True:
        begin_time = time.time()
        _, frame = cap.read()

        # scale_percent = 100
        # w = int(frame.shape[1] * scale_percent / 100)
        # h = int(frame.shape[0] * scale_percent / 100)

        # Processsing image resized to 320 X 240
        org_h, org_w = frame.shape[:2]
        w, h = 320, 240 
        dim = (w, h)
        scaling = org_w // w
        print("-" * 50)
        print(f"Image downsampled by factor {scaling}")
        
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        detections = detect_face.detect_one(model, frame, device)

        # If no face is detected don't do anything
        if len(detections) == 0:
            print("No face detected sorry")
            continue
        
        if frame_num < 3:
            print("waiting for camera to get ready")
            frame_num += 1
            continue
        
        # get embeddings of detected faces
        embeddings= []
        for det in detections:
            det = det.astype(int)
            boxes, _ = det[:4], det[4]
            cropped_img = frame[boxes[1]:boxes[3], boxes[0]: boxes[2], :]
            cropped_img = torch.unsqueeze(transform(cropped_img), 0)
            with torch.no_grad():
                vec = embedder(cropped_img.to(device))
                vec = vec.to("cpu")
                embeddings.append(vec)
        # Tracking the embeddings
        trackers = tracker.update(detections, embeddings=embeddings)
        # ************************************************
        # Checking time 
        execution_time = time.time() - begin_time
        fps = np.around(1/ execution_time, 2)
        # ************************************************
        

        cv2.line(frame, (w//2, 0), (w//2, h), color=(0, 0, 0), thickness=3)
        cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        for d in trackers:
            print(d)
            if int2labels[int(d[4])] == person_track:    
                cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (2, 255, 0), 1)
                cv2.putText(frame, person_track, (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                # print(int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                bb_x = int((((d[0] + d[2]) / 2) / w) * 640)
            #     continue
            # continue                
            else:
                if (frame_num % 5) == 0:
                    print("Frame num", frame_num)
                    print(int2labels[int(d[4])])
                    # print(int(d[4]))
                # cv2.putText(frame, int2labels[d[4]], (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        frame_num = frame_num + 1
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recordings
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    cap.release()


if __name__ == '__main__':
    main()
