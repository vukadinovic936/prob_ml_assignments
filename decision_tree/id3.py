from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
print('Classes to predict: ', data.target_names)

X = data.data
y = data.target
print('Number of examples in the data"', X.shape[0])

X_train, X_test, y_train, y_test, = train_test_split(X, y, 
                                                     random_state=47,
                                                     test_size=0.25)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

print(DecisionTreeClassifier(class_weight=None,
                             criterion='entropy',
                             max_depth=None,
                             max_features=None,
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             min_samples_leaf=1,
                             min_samples_split=2,
                             min_weight_fraction_leaf=0.0,
                             presort=False,
                             random_state=None,
                             splitter='best'))

# Predicting labels on the test set.
y_pred = clf.predict(X_test)

#print('Accuracy Score on train data: ',
#      accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))

#print('Accuracy Score on test data: ',
#      accuracy_score(y_true=y_test, y_pred=y_pred))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
clf.fit(X_train, y_train)

print('Accuracy Score on train data: ',
      accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))

print('Accuracy Score on the test data: ',
      accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
