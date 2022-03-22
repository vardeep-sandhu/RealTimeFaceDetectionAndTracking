import cupy as xp 
import sklearn.model_selection
from sklearn.datasets import load_digits
from thundersvm import SVC

# Load the digits dataset, made up of 1797 8x8 images of hand-written digits
digits = load_digits()

# Divide the data into train, test sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target)

# Move data to GPU
x_train = xp.asarray(x_train)
x_test =  xp.asarray(x_test)
y_train = xp.asarray(y_train)
y_test = xp.asarray(y_test)

# initialize a new predictor
# svm = SVC(kernel='rbf', kernel_params={'sigma': 15}, classification_strategy='ovr', x=x_train, y=y_train, n_folds=3, use_optimal_lambda=True, display_plots=True)
svm = SVC()
svm.fit(x_train, y_train)

# predict things

svm.predict(x_test)

# compute misclassification error
misclassification_error = svm.compute_misclassification_error(x_test, y_test)
print('Misclassification error, lambda = {} : {}\n'.format(svm._lambduh, misclassification_error))
