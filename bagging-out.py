from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
 DecisionTreeClassifier(), n_estimators=500,
 bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

#0.93066666666666664 : near about 93.1% accuracy

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
#93.6%