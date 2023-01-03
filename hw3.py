from sklearn import datasets

# data loading using sklearn
wine = datasets.load_wine()
X = wine.data
y = wine.target

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# splitting data
from sklearn.model_selection import train_test_split
# splits training (85%) from testing (15%)
X_trainDev, X_test, y_trainDev, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)
# splits training (85%) from development (15%)
X_train, X_dev, y_train, y_dev = train_test_split(X_trainDev, y_trainDev, test_size=0.3, random_state=1)


# SVM with default hyperparameters
from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=0)      # Support Vector Machines hyperparameter designation
svm.fit(X_train, y_train)                       # Default SVM training

y_dev_preds_SVM = svm.predict(X_dev)            # Default SVM Predicting on Dev data

# Default SVM accuracy calculation
correct1 = 0
for i in range(len(y_dev)):
    if y_dev[i] == y_dev_preds_SVM[i]:
        correct1 += 1
print('Development Accuracy Results\nDefault SVM Accuracy:' + str(correct1/len(y_dev)))


# SVM with altered hyperparameters
altered_svm = SVC(kernel='linear', C=0.01, gamma=0.001, random_state=0)         # SVM with altered hyperparams
altered_svm.fit(X_train, y_train)                                               # altered SVM training

y_dev_preds_SVM_alt = altered_svm.predict(X_dev)                                # altered SVM prediction

# Altered SVM accuracy calculation
correct2 = 0
for i in range(len(y_dev)):
    if y_dev[i] == y_dev_preds_SVM_alt[i]:
        correct2 += 1
print('Altered SVM Accuracy:' + str(correct2/len(y_dev)))


# KNN Algorithm from scratch
from math import sqrt
from collections import Counter


def knn(data, labels, query, k):
    # knn algoritm, requires 4 inputs:
    #   data = dataset without labels
    #   labels = dataset labels (with same indices)
    #   query = row (point) being tested
    #   k = the number of neighbors being taken into account for label prediction
    distances = []

    # formula used for calculating distance between two rows (points)
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):              # iterates through row features
            distance += (row1[i] - row2[i]) ** 2    # used Euclidean formula to determine distance
        return sqrt(distance)

    index = 0                                           # tracks current index
    for example in data:
        distance = euclidean_distance(example, query)   # calculates dist between query point and current point in data
        distances.append((distance, index))             # adds (distance, index) to list
        index += 1
        sorted_distances = sorted(distances)            # sorts list so closest are first

    k_nearest_dists = sorted_distances[:k]              # takes k closest distances for prediction
    k_nearest_labels = [labels[i] for distance, i in k_nearest_dists]       # finds labels of selected k closest points
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]     # makes prediction based on most common label
    return prediction


# Runs KNN algorithm for every row in Dev data
predicted_labs = []
curr_data = X_dev
curr_labels = y_dev
for a in curr_data:                                         # iterates through dataset so every point is query once
    predicted_labs.append(knn(X_train, y_train, a, k=7))    # predicts label (this is where the k value is changed)
                                                            # k=9 gives best results in training
# KNN accuracy on Development data
correct = 0
for j in range(len(predicted_labs)):
    if predicted_labs[j] == curr_labels[j]:
        correct += 1
print('KNN Accuracy: %s' % (str(correct/len(curr_labels))))       # assess accuracy of predictions


# Baseline accuracy for dev data
from sklearn.dummy import DummyClassifier
dummy_mf = DummyClassifier(strategy="most_frequent")
dummy_mf.fit(X_train, y_train)                          # training dummy model #1
dummy_mf.predict(X_dev)
print('Most Frequent Baseline: ' + str(dummy_mf.score(X_dev, y_dev)))   # dummy #1 model accuracy

dummy_st = DummyClassifier(strategy="stratified", random_state=0)
dummy_st.fit(X_train, y_train)                          # training dummy model #2
dummy_st.predict(X_dev)
print('Stratified Baseline: ' + str(dummy_st.score(X_dev, y_dev)))      # dummy #2 model accuracy


# Baseline tests for test data
from sklearn.dummy import DummyClassifier
dummy_mf = DummyClassifier(strategy="most_frequent")
dummy_mf.fit(X_train, y_train)
dummy_mf.predict(X_test)
print('\n\nTest Accuracy Results\nMost Frequent Test Baseline: ' + str(dummy_mf.score(X_test, y_test)))

dummy_st = DummyClassifier(strategy="stratified", random_state=0)
dummy_st.fit(X_train, y_train)
dummy_st.predict(X_test)
print('Stratified Baseline: ' + str(dummy_st.score(X_test, y_test)))


# altered SVM on Test Data
y_test_preds_SVM_alt = altered_svm.predict(X_test)

correct3 = 0
for i in range(len(y_test)):
    if y_test[i] == y_test_preds_SVM_alt[i]:
        correct3 += 1
print('Altered SVM Accuracy:' + str(correct3/len(y_test)))


# KNN on Test Data
predicted_labs2 = []
for a in X_test:
    predicted_labs2.append(knn(X_train, y_train, a, k=9))

correct4 = 0
for j in range(len(predicted_labs2)):
    if predicted_labs2[j] == y_test[j]:
        correct4 += 1
print('KNN Accuracy: %s' % (str(correct4/len(y_test))))




