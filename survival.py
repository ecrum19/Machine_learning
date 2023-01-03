import numpy as np
import pandas as pd
import os
path = os.getcwd()

training = pd.read_csv(path + '/titanic/train.csv')
outcome_key = pd.read_csv(path + '/titanic/gender_submission.csv')
testing = pd.read_csv(path + '/titanic/test.csv')


# Training method
def identify_weights(train_data):
    outcome = train_data['Survived'].to_numpy()     # imports all features as separate variables
    pclass = train_data['Pclass'].to_numpy()
    gender = train_data['Sex'].to_numpy()
    age = train_data['Age'].to_numpy()
    SibSp = train_data['SibSp'].to_numpy()
    parch = train_data['Parch'].to_numpy()
    embarked = train_data['Embarked'].to_numpy()

    lived = []
    died = []
    for i in range(len(training) - 1):      # tells position of those that lived / died
        if outcome[i] == 1:
            lived.append(i)
        elif outcome[i] == 0:
            died.append(i)

# Lived data processing
    lived_pclass = {}
    lived_gender = {}
    lived_SibSp = {}
    lived_parch = {}
    lived_embarked = {}
    lived_age = {}

    # makes age categorical
    lived_age['Young (0-20)'] = 0
    lived_age['Middle Aged (21-40)'] = 0
    lived_age['Older (41+)'] = 0

    # make age categorical
    for k in lived:
        if age[k] < 21:
            lived_age['Young (0-20)'] += 1
        elif age[k] < 41:
            lived_age['Middle Aged (21-40)'] += 1
        else:
            lived_age['Older (41+)'] += 1

    # Determining overall lived numbers for statistical calculations
    for j in lived:
        try:
            if lived_pclass[pclass[j]]:
                lived_pclass[pclass[j]] = lived_pclass[pclass[j]] + 1
            if lived_gender[gender[j]]:
                lived_gender[gender[j]] = lived_gender[gender[j]] + 1
            if lived_SibSp[SibSp[j]]:
                lived_SibSp[SibSp[j]] = lived_SibSp[SibSp[j]] + 1
            if lived_parch[parch[j]]:
                lived_parch[parch[j]] = lived_parch[parch[j]] + 1
            if lived_embarked[embarked[j]]:
                lived_embarked[embarked[j]] = lived_embarked[embarked[j]] + 1
        except:
            lived_pclass[pclass[j]] = 1
            lived_gender[gender[j]] = 1
            lived_SibSp[SibSp[j]] = 1
            lived_parch[parch[j]] = 1
            lived_embarked[embarked[j]] = 1


# Died data processing
    died_pclass = {}
    died_gender = {}
    died_SibSp = {}
    died_parch = {}
    died_embarked = {}
    died_age = {}

    # makes age categorical
    died_age['Young (0-20)'] = 0
    died_age['Middle Aged (21-40)'] = 0
    died_age['Older (41+)'] = 0

    for k in died:
        if age[k] < 21:
            died_age['Young (0-20)'] += 1
        elif age[k] < 41:
            died_age['Middle Aged (21-40)'] += 1
        else:
            died_age['Older (41+)'] += 1

    # Determining overall died numbers for statistical calculations
    for h in died:
        try:
            if died_pclass[pclass[h]]:
                died_pclass[pclass[h]] = died_pclass[pclass[h]] + 1
            if died_gender[gender[h]]:
                died_gender[gender[h]] = died_gender[gender[h]] + 1
            if died_SibSp[SibSp[h]]:
                died_SibSp[SibSp[h]] = died_SibSp[SibSp[h]] + 1
            if died_parch[parch[h]]:
                died_parch[parch[h]] = died_parch[parch[h]] + 1
            if died_embarked[embarked[h]]:
                died_embarked[embarked[h]] = died_embarked[embarked[h]] + 1
        except:
            died_pclass[pclass[h]] = 1
            died_gender[gender[h]] = 1
            died_SibSp[SibSp[h]] = 1
            died_parch[parch[h]] = 1
            died_embarked[embarked[h]] = 1


# statistics to determine weights
    # ratios to determine the information more likely to determine lived
    pclass_ratios = {}          # lived / died
    gender_ratios = {}
    SibSp_ratios = {}
    parch_ratios = {}
    embarked_ratios = {}
    age_ratios = {}
    for a in lived_pclass:
        pclass_ratios[a] = lived_pclass[a]/died_pclass[a]
    for b in lived_gender:
        gender_ratios[b] = lived_gender[b]/died_gender[b]
    for c in lived_SibSp:
        SibSp_ratios[c] = lived_SibSp[c]/died_SibSp[c]
    for d in lived_parch:
        parch_ratios[d] = lived_parch[d]/died_parch[d]
    for e in died_embarked:         # avoids 'nan' attribute present in lived_embarked
        embarked_ratios[e] = lived_embarked[e]/died_embarked[e]
    for f in lived_age:
        age_ratios[f] = lived_age[f]/died_age[f]

    # determine weights + best responses to use for predictions
    ratios = [pclass_ratios, SibSp_ratios, parch_ratios, embarked_ratios, age_ratios, gender_ratios]
    vals = []
    for r in ratios:
        for g in r:
            vals.append(r[g])
    vs = sorted(vals)

    h1 = vs[-1]
    h2 = vs[-2]
    h3 = vs[-3]
    h4 = vs[-4]

    # Weight calculations
    h1_weight = h1/(h1+h2+h3+h4)
    h2_weight = h2/(h1+h2+h3+h4)
    h3_weight = h3/(h1+h2+h3+h4)
    h4_weight = h4/(h1+h2+h3+h4)

    return [h1_weight, h2_weight, h3_weight, h4_weight]


# Testing method
def prediction(weights, data, key):
    # data processing in same manner as training
    outcomes_t = key['Survived']
    pclass_t = data['Pclass'].to_numpy()
    gender_t = data['Sex'].to_numpy()
    SibSp_t = data['SibSp'].to_numpy()
    parch_t = data['Parch'].to_numpy()

    # makes predictions using the weights determined in training
    predictions = []
    for row in range(1, len(data)):
        pred_score = 0

        # adds weight if present, subtracts weight if absent
        if gender_t[row] == 'female':
            pred_score += weights[0]
        else:
            pred_score -= weights[0]

        if SibSp_t[row] == 1:
            pred_score += weights[1]
        else:
            pred_score -= weights[1]

        if pclass_t[row] == 1:
            pred_score += weights[2]
        else:
            pred_score -= weights[2]

        if parch_t[row] == 3:
            pred_score += weights[3]
        else:
            pred_score -= weights[3]

        if pred_score > 0:
            predictions.append(1)
        else:
            predictions.append(0)

    # loads testing outcome data for accuracy calculation
    actual = []
    for y in range(len(outcomes_t)):
        actual.append(outcomes_t[y])

    # checks if passengers were accurately classified
    correct = 0
    incorrect = 0
    false_positive = 0      # predicted to live but actually died
    false_negative = 0      # predicted to die but actually lived
    for k in range(1, len(outcomes_t)):
        if outcomes_t[k] == predictions[k-1]:
            correct += 1
        else:
            if outcomes_t[k] > predictions[k-1]:
                false_negative += 1
            else:
                false_positive += 1
            incorrect += 1

    # prints accuracy assessment

    print('Accuracy: %.3f' % (correct/(correct+incorrect)))
    print('Number of False Positives: %d' % false_positive)
    print('Number of False Negatives: %d' % false_negative)



# main
prediction(identify_weights(training), testing, outcome_key)
