import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train_data = pd.read_csv('train.csv')
train_data.head()

test_data = pd.read_csv('test.csv')
test_data.head()

#check gender criteria for determining survival
women = train_data.loc[train_data.Sex == 'female']['Survived']
women_who_survived = (sum(women)/len(women)) * 100
print("{}% of women survived the disaster.".format(round(women_who_survived, 4)))

men = train_data.loc[train_data.Sex == 'male']['Survived']
men_who_survived = (sum(men)/len(men)) * 100
print("{}% of men survived the disaster.".format(round(men_who_survived, 4)))

from sklearn.ensemble import RandomForestClassifier as rfc

y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
x = pd.get_dummies(train_data[features])
x_test = pd.get_dummies(test_data[features])

model = rfc(n_estimators=100, max_depth=5, random_state=1)
model.fit(x,y)
predictions = model.predict(x_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Submission saved successfully!")