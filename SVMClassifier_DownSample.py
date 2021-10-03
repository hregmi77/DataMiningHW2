import pandas as pd
import arff
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, roc_curve, auc
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

# Read Data
traindata = pd.read_csv("CombinedTrain10000_Filtered.csv")
train_cols = traindata.columns
# Reading and Filtering Testdata
testdatafull = pd.read_csv("CombinedTest10000_Normalized.csv")
testdata = testdatafull[train_cols]
test_x = testdata.drop(columns='334')
test_y = testdata['334']
# Reading Train data
train_x = traindata.drop(columns='334')
train_x = np.array(train_x)
train_y = traindata['334']
train_y = np.array(train_y)
# Balancing the dataset
train_bx, train_by = balanced_subsample(train_x, train_y)
clf = svm.SVC()
clf.fit(train_bx, train_by)
scores = cross_val_score(clf, train_bx, train_by, cv=10)
predicted_y = clf.predict(test_x)
# Save Predicted Labels
predicted_data = pd.DataFrame(predicted_y)
predicted_data.to_csv("Predicted_SVM.csv")
# Compute Accuracy
print('Accuracy:', clf.score(test_x, test_y))
# Get and print Scores
scores_svm = precision_recall_fscore_support(test_y, predicted_y, average='binary')
auc_svm = roc_auc_score(test_y, predicted_y)
mcc_svm = matthews_corrcoef(test_y, predicted_y)
print('Precision',scores_svm[0], 'Recall', scores_svm[1], 'MCC', mcc_svm, 'ROC Area', auc_svm)

# Result is on the train data
predicted_y = clf.predict(train_bx)
# Save Predicted Labels
predicted_data = pd.DataFrame(predicted_y)
predicted_data.to_csv("Predicted_SVM_Train.csv")
# Compute Accuracy
print('Accuracy:', clf.score(train_bx, train_by))
# Get and print Scores
scores_svm = precision_recall_fscore_support(train_by, predicted_y, average='binary')
auc_svm = roc_auc_score(train_by, predicted_y)
mcc_svm = matthews_corrcoef(train_by, predicted_y)
print('Train Precision',scores_svm[0], 'Train Recall', scores_svm[1], 'Train MCC', mcc_svm, 'Train ROC Area', auc_svm)