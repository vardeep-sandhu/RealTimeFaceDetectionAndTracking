import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

def load_pickle(path):
    with open(path , 'rb') as f:
        return pickle.load(f)

def process_dataset(classes_list, embd_path):
    x, y = [], []
    for label, embd_cls in classes_list:
        if embd_cls == "Getting_Labels":
            continue
        embds = np.load(os.path.join(embd_path, f"{embd_cls}.npz"))['arr_0']
        x.append(embds)
        y.append(np.full((len(embds)), int(label)))
    return np.concatenate(x), np.concatenate(y)

def train_classifier(X_train, y_train):
    # clf = SVC(kernel="linear", C=0.025, probability=True)
    clf = SVC(gamma=2, C=1, probability=True)
    clf.fit(X_train, y_train)
    return clf

def save_classifier(classifier, path):
    with open(path, "wb") as open_file:
        pickle.dump(classifier, open_file)


def main():
    print("Making classifier")
    embd_path = "dataset/embeddings"
    classes_list = np.loadtxt("classes.txt", dtype=str)
    
    X, y = process_dataset(classes_list, embd_path)

    # Sanity Check
    assert len(X) == len(y), "Shape does not match"

    # X, y = np.squeeze(X), np.squeeze(y)
    X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.001, random_state=42
    )

    assert 0 not in np.unique(y_train), "Something is wrong"
    
    clf = train_classifier(X_train, y_train)
    save_classifier(clf, "classifier.pkl")
    print("Saving Classifier")
    print("-" * 50)

if __name__ == "__main__":
    main()

