import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

path = os.getcwd()

# loading ls dataset
own_dataset_ls = pd.read_csv(path+'/HW_dataset_ls.csv')
own_dataset_ls_labs = own_dataset_ls[['State']]
own_dataset_ls_nolabs = own_dataset_ls[['v1', 'v2']]

# loading non-ls dataset
own_dataset_nls = pd.read_csv(path+'/HW_dataset_nonls.csv')
own_dataset_nls_labs = own_dataset_nls[['State']]
own_dataset_nls_nolabs = own_dataset_nls[['v1', 'v2']]


# perceptron algorithm
class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        # Parameters -- eta (learning rate), n_iter (number of learning iterations)
        self.eta = eta
        self.n_iter = n_iter

    def fit_model(self, dataset, labels):
        # inputs -- dataset (pandas dataframe containing values for predicting label),
        #           labels (the labels for each row of the dataset)

        self.wf = np.zeros(1 + dataset.shape[1])        # initialize the weight array
        self.error_tracker = []                         # initialize the error tracker

        for l in range(self.n_iter):                # iterate over the number of iterations
            errors = 0

            for index, row in dataset.iterrows():   # iterate over the rows of the dataset

                # update value depends on the prediction of the algorithm given current weights
                update = self.eta * (labels.loc[index] - self.prediction(row))

                self.wf[1:] += update * row     # weight array update if the label did not match (dw = z * row(i))
                self.wf[0] += update            # updates threshold value with weight update
                errors += int(update != 0.0)    # if weight changes there was an error
            self.error_tracker.append(errors)   # tracks number of errors
        return self

    def prediction(self, row):
        # predicts the label via dot product of given weight array and current row x values

        activation = np.dot(row, self.wf[1:]) + self.wf[0]

        # returns label determined by dot product
        return np.where(activation >= 0.0, 1, -1)


# LS Dataset imaging
def ls_data():
    plt.scatter(own_dataset_ls_nolabs['v1'][:5], own_dataset_ls_nolabs['v2'][:5], color='red', marker='o', label='Status=1')
    plt.scatter(own_dataset_ls_nolabs['v1'][5:], own_dataset_ls_nolabs['v2'][5:], color='blue', marker='x', label='Status=2')
    plt.show()

# non-LS Dataset imaging
def nls_data():
    plt.scatter(own_dataset_nls_nolabs['v1'][:5], own_dataset_nls_nolabs['v2'][:5], color='red', marker='o', label='Status=1')
    plt.scatter(own_dataset_nls_nolabs['v1'][5:], own_dataset_nls_nolabs['v2'][5:], color='blue', marker='x', label='Status=2')
    plt.show()


# LS data Perceptron imaging
def ls_percep():
    ls_run = Perceptron(eta=0.01, n_iter=10)
    ls_run.fit_model(own_dataset_ls_nolabs, own_dataset_ls_labs)
    plt.plot(range(1, len(ls_run.error_tracker) + 1), ls_run.error_tracker, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    print(ls_run.error_tracker[-1] / 10)
    plt.show()


# non-ls data Perceptron imaging
def nls_percep():
    nls_run = Perceptron(eta=0.01, n_iter=10)
    nls_run.fit_model(own_dataset_nls_nolabs, own_dataset_nls_labs)
    plt.plot(range(1, len(nls_run.error_tracker) + 1), nls_run.error_tracker, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    print(nls_run.error_tracker[-1] / 10)
    plt.show()

# Driver code
#ls_data()
#nls_data()
#ls_percep()
#nls_percep()

# loading titanic dataset
training = pd.read_csv(path + '/titanic/train.csv')
whole_dataset = training

# make categorical data into numeric data using pd.categorize()
sex_codes, sex_uniques = pd.factorize(whole_dataset['Sex'])             # changes sex to numeric
whole_dataset['Sex'] = sex_codes
name_codes, name_uniques = pd.factorize(whole_dataset['Name'])          # changes name to numeric
whole_dataset['Name'] = name_codes
ticket_codes, ticket_uniques = pd.factorize(whole_dataset['Ticket'])    # changes ticked to numeric
whole_dataset['Ticket'] = ticket_codes
cabin_codes, cabin_uniques = pd.factorize(whole_dataset['Cabin'])       # changes cabin to numeric
whole_dataset['Cabin'] = cabin_codes
embarked_codes, embarked_uniques = pd.factorize(whole_dataset['Embarked'])  # changes embarked to numeric
whole_dataset['Embarked'] = embarked_codes
age_codes, age_uniques = pd.factorize(whole_dataset['Age'])  # changes embarked to numeric
whole_dataset['Age'] = age_codes
pclass_codes, pclass_uniques = pd.factorize(whole_dataset['Pclass'])  # changes embarked to numeric
whole_dataset['Pcalss'] = pclass_codes
SibSp_codes, SibSp_uniques = pd.factorize(whole_dataset['SibSp'])  # changes embarked to numeric
whole_dataset['SibSp'] = SibSp_codes
Parch_codes, Parch_uniques = pd.factorize(whole_dataset['Parch'])  # changes embarked to numeric
whole_dataset['Parch'] = Parch_codes

# randomly splitting the titanic dataset into 70% training and 30% testing via pandas sample()
training_data = whole_dataset.sample(frac=0.7, random_state=1)
testing_data = whole_dataset.drop(training_data.index)

# splitting labels and rest of dataset for training
training_data['Survived'] = training_data['Survived'].replace([0],-1)
training_data_labs = training_data['Survived']
training_data_nolabs = training_data.drop('Survived', axis=1)


# splitting labels and rest of dataset for testing
testing_data['Survived'] = testing_data['Survived'].replace([0],-1)
testing_data_labs = testing_data['Survived']
testing_data_nolabs = testing_data.drop('Survived', axis=1)


# batch adalineGD algorithm from book
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.000001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression, we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Testing to find the most predictive features
#data_1 = training_data.drop(['Sex', 'PassengerId', 'Name', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Pcalss', 'Fare'], axis=1)
#data_1_test = testing_data.drop(['Sex','PassengerId', 'Name', 'Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Pcalss', 'Fare'], axis=1)
data_1 = training_data.drop(['Survived', 'Ticket'], axis=1)
data_1_test = testing_data.drop(['Survived', 'Ticket'], axis=1)


# driver for training
ada_train = AdalineGD(n_iter=10, eta=0.000000001).fit(data_1, training_data_labs)


def ada_image():
    # makes image of Adaline training results
    plt.plot(range(1, len(ada_train.cost_) + 1), ada_train.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.title('Adaline - Learning rate 0.000000001')
    plt.show()

# driver
#ada_image()


# Baseline model
class Baseline(object):
    def __init__(self, random_seed=1):
        self.random_seed = random_seed

    def test_model(self, dataset, labels):
        # inputs -- dataset (pandas dataframe containing values for predicting label),
        #           labels (the labels for each row of the dataset)

        rgen = np.random.RandomState(self.random_seed)
        self.wf = rgen.normal(loc=0.0, scale=0.01, size=1 + dataset.shape[1])  # initialize weight array with random numbers

        errors = 0
        for index, row in dataset.iterrows():  # iterate over the rows of the dataset

            # update value depends on the prediction of the algorithm given current weights
            update = labels.loc[index] - self.prediction(row)
            errors += int(update != 0.0)  # if weight changes there was an error

        print('Baseline accuracy: ' + str(errors/len(dataset)))
        return self

    def prediction(self, row):
        # predicts the label via dot product of given weight array and current row x values
        activation = np.dot(row, self.wf[1:]) + self.wf[0]
        # returns label determined by dot product
        return np.where(activation >= 0.0, 1, -1)


# Tests perceptron + adaline model performance via final training weight values and test dataset
def model_tests(model, f_weights, dataset, labels):
    errors = 0

    def prediction(row):
        # outputs activation value via dot product of given weight array and current row x values
        activation = np.dot(row, f_weights[1:]) + f_weights[0]
        # returns label determined via the activation value compared to theta
        return np.where(activation >= f_weights[0], 1, -1)

    for index, row in dataset.iterrows():  # iterate over the rows of the dataset
        # update value depends on the prediction of the algorithm given current weights
        update = labels.loc[index] - prediction(row)
        errors += int(update != 0.0)        # if weight changes there was an error
    print('%s accuracy: ' % model + str(1-(errors / len(dataset))))     # prints accuracy


def testing():
    # calls testing functions
    Baseline().test_model(training_data_nolabs, training_data_labs)
    perceptron_train = Perceptron(eta=0.0001, n_iter=10).fit_model(training_data_nolabs, training_data_labs)
    model_tests('Perceptron', perceptron_train.wf, testing_data_nolabs, testing_data_labs)
    model_tests('AdalineGD', ada_train.w_, data_1_test, testing_data_labs)


# Driver code
testing()




