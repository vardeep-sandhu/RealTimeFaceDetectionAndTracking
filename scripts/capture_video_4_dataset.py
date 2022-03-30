from numpy import full
import cv2
import os
import argparse
import shutil
import detect_face 
import torch

def capture_vid(name, vid_dir):
    print("Capturing Video")
    print("Video saved at", vid_dir)
    video = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if (video.isOpened() == False): 
      print("Error reading video file")
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    out = cv2.VideoWriter(os.path.join(vid_dir, f'{name}.avi'), fourcc, 10.0, size)
    count = 0
    while(video.isOpened()):
        ret, frame = video.read()
        count += 1
        if ret==True:
            out.write(frame)
            cv2.imshow('frame',frame)
            # 30 seconds works good for us. Since ideally the system performs best when the detected faces are 500-600.
            # This also keeps the dataset balanced, since all other classes have 500-600 images  
            
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (count == 650):
                print("Video Capture ended")
                print("-" * 50)
                break
        else:
            break
        
    video.release()
    out.release()

    cv2.destroyAllWindows()

def get_frames_frm_vid(name, dim, vid_dir, full_photo_dir):
    print("Getting frames from Video")
    print("Size of saved images is:", dim)
    
    vid_path = os.path.join(vid_dir, f"{name}.avi")
    write_path = os.path.join(full_photo_dir, name)

    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    label = os.path.basename(write_path)
    count = 0
    if not os.path.isdir(write_path):
        os.makedirs(write_path)

    while success:
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"{write_path}/{label}{count}.jpg", image)     # save frame as JPEG file      
        success, image = vidcap.read()
        if count == 650:
            break
        count += 1
    
    print(f"Got {count+1} frames from video")
    print("-" * 50)


def get_faces(name, dim, faces_dir, full_photo_dir):
    print("Getting faces from these images")
    model_path = "weights/face.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = detect_face.load_model(model_path, device)
    
    save_dir = os.path.join(faces_dir, name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    input_dir = os.path.join(full_photo_dir, name)
    files = os.listdir(input_dir)
    count = 0
    for file in files:
        frame = cv2.imread(os.path.join(input_dir, file))
        h, w = frame.shape[:2]
        
        # Sanity check
        assert dim == (w, h), "Dimension issue, Check again!"
        
        dets = detect_face.detect_one(model, frame, device)

        # Length of detection has to be 1, if 2 faces in the training dataset then image ignored
        if len(dets) != 1:
            continue

        dets = dets.astype(int)
        boxes, _ = dets[0][:4], dets[0][4]

        # Crop 10 more pixels to get more of the faceial information
        cropped_img = frame[boxes[1]-10:boxes[3]+10, boxes[0]-10: boxes[2]+10, :]
        count += 1
        cv2.imwrite(os.path.join(save_dir, file), cropped_img)
    
    print(f"Gotten {count + 1 } faces from all images ")
    print("-" * 50)

def clear_old_faces(new_class, labels, faces_dir, full_photos_dir):
    labels.append(new_class)

    present_classes = os.listdir(faces_dir)

    for cls in present_classes:
        if cls not in labels:
            rem_dir_faces = os.path.join(faces_dir, cls)
            shutil.rmtree(rem_dir_faces)


def arg_parser():
    parser = argparse.ArgumentParser(description="add, something")
    parser.add_argument('--name', '-n', help="Give the name of existing video file or the name of the dataset face")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    # Directory paths
    vid_dir = "dataset/video/"
    full_photo_dir = "dataset/full_photos"
    faces_dir = "dataset/faces"
    w, h = 320, 240
    dim = (w, h)
    labels = ['Migel', 'Vardeep', 'Leticia', 'Masih']

    if args.name is not None:
        capture_vid(args.name, vid_dir)
        get_frames_frm_vid(args.name, dim, vid_dir, full_photo_dir)
        get_faces(args.name, dim, faces_dir, full_photo_dir)
        
    clear_old_faces(args.name, labels, faces_dir, full_photo_dir)
