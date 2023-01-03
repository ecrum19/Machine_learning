from sklearn import datasets
from random import randint, randrange, seed
from sklearn.svm import SVC
import numpy as np

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
# splits training (85%) from testing (15%) (and shuffels data)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)


# SVM training using cross-fold validation
# randomly split the training set into n parts and repeatedly train on n-1 parts and test on the remaining part
def cross_fold_svm(Xd, yd):
    def test(model, dev_data_x, dev_data_y, current_fold):        # Tests SVM on input development data
        y_test_pred = model.predict(dev_data_x)
        correct = 0
        for i in range(len(dev_data_y)):
            if y_test_pred[i] == dev_data_y[i]:
                correct += 1
        accuracy = correct/len(dev_data_y)
        print('Fold %d SVM Accuracy: %f' % (current_fold, accuracy))
        return accuracy

    def train(train_data_x, train_data_y):      # Trains SVM on input training data
        seed(1)
        svm = SVC(kernel='linear', random_state=0)
        svm.fit(train_data_x, train_data_y)
        return svm

    def split(dataset_x, dataset_y, num_folds):     # splits input data into designated number of folds
        dataset_split_x = []
        dataset_split_y = []
        dfx_copy = dataset_x
        dfy_copy = dataset_y
        fold_size = int(dataset_x.shape[0] / num_folds)     # determines how large folds are given number of folds

        for _ in range(num_folds):
            foldx = []
            foldy = []
            while len(foldx) < fold_size:               # while loop to add elements to the folds
                r = randint(0, dfx_copy.shape[0] - 1)   # select a random row from data
                foldx.append(dfx_copy[r])               # save the randomly selected line to current fold
                foldy.append(dfy_copy[r])
                dfx_copy = np.delete(dfx_copy, r, 0)    # delete the randomly selected line so not added twice
                dfy_copy = np.delete(dfy_copy, r, 0)

            dataset_split_x.append(np.asarray(foldx))   # fold is added to list of folds
            dataset_split_y.append(np.asarray(foldy))
        return dataset_split_x, dataset_split_y

    n = randint(0, int(len(X_train) / 2))       # random n determines number of folds
    currdata_x, currdata_y = split(Xd, yd, n)
    errors = []
    curr_dev = 0
    for curr_f in range(n):             # loops over number of folds
        poss_folds = list(range(0, n))
        poss_folds.remove(curr_dev)     # removes fold that is acting as development fold from training data
        dev_x = currdata_x[curr_dev]    # assigns development X and Y data
        dev_y = currdata_y[curr_dev]
        curr_dev += 1
        for next_fold in poss_folds:            # concatenates non-development data for training
            if next_fold == poss_folds[0]:
                cv_x = currdata_x[next_fold]    # first fold added to training data
                cv_y = currdata_y[next_fold]
            else:
                cv_x = np.concatenate((cv_x, currdata_x[next_fold]), axis=0)    # subsequent folds added to training
                cv_y = np.concatenate((cv_y, currdata_y[next_fold]), axis=0)

        svm_mod = train(cv_x, cv_y)     # calls train method to train SVM on current training data
        errors.append(test(svm_mod, dev_x, dev_y, curr_dev))      # calls test method to test SVM on curr dev data

    print("Average Accuracy over %d-folds: %f" % (n, (sum(errors) / len(errors))))


cross_fold_svm(X_train, y_train)


# Grid Search Algorithm
def grid_search():
    X_t, X_d, y_t, y_d = train_test_split(X_train, y_train, test_size=0.3, random_state=0)  # Split dataset into train and dev set

    max_accuracy = 0.0      # tracks best model hyperparameter combinations
    best_c = 0
    best_gamma = 0

    # grid searching for learning rate
    c = [.00001, .0001, .001, .01, .1, 1, 10, 100, 10000]           # determines possible values from C and gamma
    gamma = [.00001, .0001, .001, .01, .1, 1, 10, 100, 10000]

    for c_val in c:                     # nested for loop to test every C with every gamma
        for gamma_val in gamma:
            svm2 = SVC(kernel='linear', random_state=0, C=c_val, gamma=gamma_val)       # model defined with hyperparams
            svm2.fit(X_t, y_t)              # model fitting
            y_dev_pred = svm2.predict(X_d)  # model predictions
            correct = 0
            for g in range(len(y_dev_pred)):    # determine accuracy of model
                if y_dev_pred[g] == y_d[g]:
                    correct += 1
            accuracy = correct / len(y_d)

            if accuracy > max_accuracy:     # tracks the most accurate combination of hyperparams
                max_accuracy = accuracy
                best_c = c_val
                best_gamma = gamma_val

    print("\nHighest Accuracy Found via Grid Search: %f\n\tC-value: %f\n\tGamma value: %f" % (max_accuracy, best_c, best_gamma))


    # Testing
    test_correct = 0
    svm3 = SVC(kernel='linear', random_state=0, C=0.01, gamma=0.00001)
    svm3.fit(X_train, y_train)
    y_test_pred = svm3.predict(X_test)
    for s in range(len(y_test)):            # determine accuracy
        if y_test_pred[s] == y_test[s]:
            test_correct += 1
    test_accuracy = test_correct / len(y_test)
    print("\nOptimized SVM Test Data Accuracy: %f" % test_accuracy)

    # Training
    train_correct = 0
    svm4 = SVC(kernel='linear', random_state=0, C=0.01, gamma=0.00001)
    svm4.fit(X_train, y_train)
    y_train_pred = svm4.predict(X_train)
    for s in range(len(y_train)):           # determine accuracy
        if y_train_pred[s] == y_train[s]:
            train_correct += 1
    train_accuracy = train_correct / len(y_train)
    print("Optimized SVM Train Data Accuracy: %f" % train_accuracy)


grid_search()






