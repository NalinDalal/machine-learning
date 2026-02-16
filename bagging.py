from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, #500 decision tree classifiers
    max_samples=100, bootstrap=True, n_jobs=-1  #100 training instances; n_jobs: number of CPU cores to use for training and predictions
    #-1 means all
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
