import os
import torch
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from dataset import Custom_Dataset
import numpy as np

def testing(test_loader):
    with torch.no_grad():
        embeddings = []
        for img in test_loader:
            img = img.to(device)
            embds = embedder(img)
            embeddings.append(embds.to("cpu"))
    return torch.cat(embeddings)

def make_embds(test_loader, Epochs):
    all_embds = [] 
    for epoch in range(Epochs):
        # print(f"Starting Epoch: {epoch+1} of {Epochs}")
        embds = testing(test_loader)
        all_embds.append(embds)
    return torch.cat(all_embds)

def clear_old_embeddings(labels, embd_path):
    for embd in os.listdir(embd_path):
        if os.path.splitext(embd)[0] in labels:
            continue
        else:
            os.remove(os.path.join(embd_path, embd))

if __name__ == '__main__':
    
    print("Now we are getting embeddings: ")
    
    # Check if the dir exists [TODO]
    faces_dir = "dataset/faces"
    embd_path = "dataset/embeddings"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    labels = os.listdir(faces_dir)
    # labels will be a list like this:
    # labels = ['Migel', 'Vardeep', 'Leticia', 'Masih']

    clear_old_embeddings(labels, embd_path)
    int2labels = np.array([[i+1,label] for i,label in enumerate(labels)])
    int2labels = np.insert(int2labels, 0, ["0", "Getting_Labels"], axis=0)
    np.savetxt("classes.txt", int2labels, fmt='%s')
    print("Classes Saved in project dir")
    print("-" * 50)
    print(int2labels)
    curr_embds = [os.path.splitext(i)[0] for i in os.listdir(embd_path)]
    
    for cls in labels:
        if cls in curr_embds:
            continue
        new_class_dataset = Custom_Dataset(faces_dir, cls)
        new_class_loader = DataLoader(new_class_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        embds = make_embds(new_class_loader, Epochs=3)
        
        assert embds.shape[0] == len(new_class_dataset) * 3, "Number of images don't match"
        np.savez(os.path.join(embd_path, cls), embds)
        print(f"Saved Embeddings for class {cls}")
        