import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

df_results = pd.read_csv('archive/results.csv')
df_constructor_standings = pd.read_csv('archive/constructor_standings.csv')
df_driver_standings = pd.read_csv('archive/driver_standings.csv')

df_driver_standings.drop('positionText', axis=1)
df_constructor_standings.drop('positionText', axis=1)

df_merged =pd.merge(df_results,df_constructor_standings, on = ['constructorId','raceId']) # merge the dataframes
df_merged = pd.merge(df_merged,df_driver_standings, on = ['raceId','driverId'])
results = (df_merged['position_x']).replace("\\N",20)

results_cat = results
results_cat = results_cat.astype(int)


results_cat = np.where(results_cat<4,0,results_cat) #anything less than four will have 0
results_cat = np.where((results_cat >= 4) & (results_cat <=10),1,results_cat) #anything between 4-10 will be given the number 1 category
results_cat = np.where(results_cat>10,2,results_cat) #above 10 will have the label 2
results_cat = pd.DataFrame(results_cat)

df_no_results = df_merged.drop(['resultId', 'constructorStandingsId', 'position','positionText_x', 'position_x','positionText_y','positionOrder','positionText', 'points_x','time','milliseconds','fastestLapSpeed', 'fastestLapTime','points_y','position_y','statusId'], axis=1) #drop all the columns that are verry predictive (the labels)
# df_no_results['position_x'] = df_no_results['position_x'].replace('\\N',21) #replace missing possitions with 21
df_no_results['rank'] = df_no_results['rank'].replace('\\N',21) #replace missing ranks with 21
df_no_results['fastestLap'] = df_no_results['fastestLap'].replace('\\N',0) #replace missing laps with 21

s = df_no_results[df_no_results.eq("\\N").any(1)]  #check if there is any other data that has values missing
list_s = list(s.index)
df_no_results = df_no_results.drop(index=list_s,inplace=False)
results = results.drop(index=list_s,inplace=False)
results_cat = results_cat.drop(index=list_s,inplace=False)

df_no_results = df_no_results.apply(pd.to_numeric, errors='coerce')   #Force the data to get rid of infinite values or missing values. There was only once instance of a missing data in the whole dataset

sc = StandardScaler() #scale the date
sc.fit(df_no_results)
df_no_results= sc.transform(df_no_results)

results = results.apply(pd.to_numeric,errors = 'coerce')

# splits uncategorized and categorized data into training (80%) and testing (20%)
uncat_trainDev, uncat_test, r_uncat_trainDev, r_uncat_test = train_test_split(df_no_results, results, test_size=0.2, random_state=0)
cat_trainDev, cat_test, r_cat_trainDev, r_cat_test = train_test_split(df_no_results, results_cat, test_size=0.2, random_state=0)

# splits uncategorized and categorized training data into training (80%) and development (20%)
uncat_train, uncat_dev, r_uncat_train, r_uncat_dev = train_test_split(uncat_trainDev, r_uncat_trainDev, test_size=0.2, random_state=0)
cat_train, cat_dev, r_cat_train, r_cat_dev = train_test_split(cat_trainDev, r_cat_trainDev, test_size=0.2, random_state=0)




def svm_select(s,X, y, X_dev, y_dev):
    kernel = ['linear', 'rbf']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    def grid_search(kernel, C):
        max_accuracy = 0.0   # tracks best model hyperparameter combinations
        best_c = 0
        best_kernel = 0
        best_f1 = 0
        accuracy_list = []
        for k,c in itertools.product(kernel,C):  # nested for loop to test every C with every gamma
                print("C-value: %s\tKernel: %s\n" % (c, k))
                svm = SVC(kernel=k, C=c, random_state=0)
                svm.fit(X, y.values.ravel())
                y_dev_pred = svm.predict(X_dev)     # predicts labels on dev data
                print("\tF1-value: %f" % f1)
                accuracy_s = accuracy_score(y_dev_pred, y_dev)
                f1 =f1_score(y_dev_pred, y_dev, average=None)
                accuracy_list.append((accuracy_s,c,f1))
                if accuracy_s > max_accuracy:
                    max_accuracy = accuracy_s
                    best_c = c
                    best_kernel = k

        print(s+"\nHighest Accuracy Found via Grid Search: %f\n\tC-value: %f\n\tKernel: %s" % (max_accuracy, best_c, best_kernel))

    grid_search(kernel, C)

def lr_select(s,X, y, X_dev, y_dev):
    penalty = ['l2', 'none']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    def grid_search(pen, C):
        max_accuracy = 0.0   # tracks best model hyperparameter combinations
        best_c = 0
        best_pen = 0
        best_f1 = 0
        accuracy_list = []
        for p,c in itertools.product(pen,C):  # nested for loop to test every C with every gamma
                print("C-value: %s\tPenalty: %s\n" % (c, p))
                lg = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty=p, C=c, random_state=0, max_iter=500)  # model defined with hyperparams
                lg.fit(X, y.values.ravel())  # model fitting
                y_dev_pred = lg.predict(X_dev)  # model predictions
                print("\tF1-value: %f" % f1)
                accuracy_s = accuracy_score(y_dev_pred, y_dev)
                f1 =f1_score(y_dev_pred, y_dev, average=None)
                accuracy_list.append((accuracy_s,c,f1))
                if accuracy_s > max_accuracy:
                    max_accuracy = accuracy_s
                    best_c = c
                    best_pen = p
        print(s+"\nHighest Accuracy Found via Grid Search: %f\n\tC-value: %f\n\tPenalty: %s" % (max_accuracy, best_c, best_pen))

    grid_search(penalty, C)




##### UNCOMMENT this in order to run the Grid Search #########
svm_select("Uncategorized", uncat_train, r_uncat_train, uncat_dev, r_uncat_dev)
svm_select("Categorized", cat_train, r_cat_train, cat_dev, r_cat_dev)


lr_select("Categorized",cat_train, r_cat_train, cat_dev, r_cat_dev)
lr_select("Uncategorized",uncat_train, r_uncat_train, uncat_dev, r_uncat_dev)


####uncat_trainDev, uncat_test, r_uncat_trainDev, r_uncat_test
svm = SVC(kernel='linear', C=100, random_state=0)
svm.fit(uncat_trainDev, r_uncat_trainDev.values.ravel())
y_dev_pred = svm.predict(uncat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_dev_pred, r_uncat_test)
f1 = f1_score(y_dev_pred, r_uncat_test, average=None)
print("SVC uncategorize Accruacy: "+str(accuracy_s))

svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(cat_trainDev, r_cat_trainDev.values.ravel())
y_dev_pred = svm.predict(cat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_dev_pred, r_cat_test)
f1 = f1_score(y_dev_pred, r_cat_test, average=None)
print("SVC categorize Accruacy: "+str(accuracy_s))


lg = LogisticRegression(penalty='l2', C=10, random_state=0)
lg.fit(cat_trainDev, r_cat_trainDev.values.ravel())
y_dev_pred = lg.predict(cat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_dev_pred, r_cat_test)
f1 = f1_score(y_dev_pred, r_cat_test, average=None)
print("LR categorized Accruacy: "+str(accuracy_s))


lg = LogisticRegression(penalty='l2', C=0.001000, random_state=0)
lg.fit(uncat_trainDev, r_uncat_trainDev.values.ravel())
y_dev_pred = lg.predict(uncat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_dev_pred, r_uncat_test)
f1 = f1_score(y_dev_pred, r_uncat_test, average=None)
print("LR uncategorized Accruacy: "+str(accuracy_s))


#################
# the test before is done on the training_development set and test set

#the test that follows is done on the training set and test. We wanted to see if there is a difrence in accurayc

#uncat_train, uncat_dev, r_uncat_train, r_uncat_dev

svm = SVC(kernel='linear', C=100, random_state=0)
svm.fit(uncat_train, r_uncat_train.values.ravel())
y_pred = svm.predict(uncat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_pred, r_uncat_test)
f1 = f1_score(y_pred, r_uncat_test, average=None)
print("SVC uncategorize Accruacy: "+str(accuracy_s))

svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(cat_train, r_cat_train.values.ravel())
y_pred = svm.predict(cat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_pred, r_cat_test)
f1 = f1_score(y_pred, r_cat_test, average=None)
print("SVC categorize Accruacy: "+str(accuracy_s))


lg = LogisticRegression(penalty='l2', C=10, random_state=0)
lg.fit(cat_train, r_cat_train.values.ravel())
y_pred = lg.predict(cat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_pred, r_cat_test)
f1 = f1_score(y_pred, r_cat_test, average=None)
print("LR categorized Accruacy: "+str(accuracy_s))


lg = LogisticRegression(penalty='l2', C=0.001000, random_state=0)
lg.fit(uncat_train, r_uncat_train.values.ravel())
y_pred = lg.predict(uncat_test)  # predicts labels on dev data
accuracy_s = accuracy_score(y_pred, r_uncat_test)
f1 = f1_score(y_pred, r_uncat_test, average=None)
print("LR uncategorized Accruacy: "+str(accuracy_s))


### Check with a dummy classifier ###

dummy_classifier = DummyClassifier(strategy="uniform")
dummy_classifier.fit(uncat_trainDev, r_uncat_trainDev.values.ravel())
y_pred = dummy_classifier.predict(uncat_test)
accuracy_s = accuracy_score(y_pred, r_uncat_test)
print(accuracy_s)
