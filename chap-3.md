# Classification

let's start with hello-world of machine learning
there is a popular data-set `MNIST`, whenever something new is found, tested on `MNIST`

to fetch `MNIST` dataset:

```python
from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')
mnist
```

the output of SciKit DataSet is generally like a simple dictionary

```
{'COL_NAMES': ['label', 'data'],
'DESCR': 'mldata.org dataset: mnist-original',
'data': array([[0, 0, 0, ..., 0, 0, 0],
[0, 0, 0, ..., 0, 0, 0],
[0, 0, 0, ..., 0, 0, 0],
...,
[0, 0, 0, ..., 0, 0, 0],
[0, 0, 0, ..., 0, 0, 0],
[0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
'target': array([ 0., 0., 0., ..., 9., 9., 9.])}
```

take a look at this:

```python
>>> X, y = mnist["data"], mnist["target"]
>>> X.shape
(70000, 784)    # 70,000 images, and each image has 784 features.28x28 size
>>> y.shape
(70000,)
```

just grab an instance's feature vector, reshape to 28x28 array
display via `imshow()`

```python
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()
```

well u can shuffle the training dataset also

```python
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

## Binary Classifier

it's like say you wanna classify `5`, so either it is `5` or 1 or it is `not-5` or 0

```python
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)
```

let's pick a classifier: `Stochastic Gradient Descent (SGD)`
handles large datasets very efficiently

```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

# Performance Measure

basically evaluating a classifier, various performance measures are there

## Cross-Validation

repeatedly splits your dataset into different train/test partitions, trains and tests the model on each, and then averages the results.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
```

## Confusion Matrix

much better way to check accuracy and evaluate is to look at confusion matrix

`cross_val_predict()` performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions made on each test fold.

```python
cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

get confusion matrix via `confusion_matrix()`

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
cm
```

row - actual class
column - predicted class

The first row of this matrix considers non-5 images (the negative class): 53,272 of them were correctly classified as non-5s (they are called truenegatives), while the remaining 1,307 were wrongly classified as 5s (false positives).

second row considers the images of 5s (the positive class): 1,077 were wrongly classified as non-5s (false negatives/FP), while the remaining 4,344 were correctly classified as 5s (true positives/TP).

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

trivial way - only 1 prediction that too correct
precision is typically used along with another metric named recall, also called sensitivity or true positive rate
(TPR): this is the ratio of positive instances that are correctly detected by the classifier

$$
[
\text{recall}=\frac{TP}{TP+FN}
]
$$

other way to do it in python

```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_pred)
recall_score(y_train_5, y_train_pred)
```

combine both into single unit `F1`

$$
[
\text{F1}=2*\frac{precision*recall}{precision+recall}
]
$$

$$
[
\text{F1}=\frac{TP}{TP+ \frac{FN+FP}{2} }
]
$$

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_pred)
```

use cases may changes like:

- classifier to check if video is safe or not{low recall, high precision}
- classifier to detect shoplifters on surveillance images{low precision, high recall}

## Precision & Recall

u can find precision and recall from scikit via:

```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_pred)
recall_score(y_train_5, y_train_pred)
```

```
\text{F1}=\frac{2}{\frac{1}{precision}+\frac{1}{recall}}
\text{F1}=2*\frac{precision*recall}{precision+recall}
\text{F1}=\frc{TP}{TP+\frac{FN+FP}{2}}
```

to calculate F1 score do this:

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_pred)
```

unfortunately u can't have both: increasing precision reduces recall, and vice versa.
known as _precision/recall tradeoff_.

### Precision/Recall Tradeoff

In binary classification, every model makes predictions based on a **decision threshold**.
By default, classifiers like `SGDClassifier` assign an instance to the _positive class_ if its **decision score** is greater than zero. Changing this threshold shifts the balance between **precision** and **recall**.

### What Happens When You Adjust the Threshold

- **Raising the threshold** → fewer positive predictions
  → Higher **precision**, but lower **recall**
- **Lowering the threshold** → more positive predictions
  → Higher **recall**, but lower **precision**

Example:

| Threshold   | True Positives | False Positives | Precision | Recall |
| ----------- | -------------- | --------------- | --------- | ------ |
| Low (0)     | 4              | 1               | 80%       | 67%    |
| High (200k) | 3              | 0               | 100%      | 50%    |

### Visualizing the Tradeoff

You can get the decision scores for all instances and use them to calculate precision and recall for different thresholds:

```python
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Now plot them:

```python
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```

### Choosing the Right Threshold

You can directly plot **precision vs recall** to find the sweet spot:

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}
$$

The ideal threshold depends on your use case:

- For **spam detection**, aim for **high recall** (catch all spam, even with a few false alarms).
- For **medical diagnosis**, prefer **high precision** (fewer false positives).

### Example

If you want a 90% precision classifier:

```python
y_train_pred_90 = (y_scores > 70000)
```

Then evaluate:

```python
precision_score(y_train_5, y_train_pred_90)  # ≈ 0.90
recall_score(y_train_5, y_train_pred_90)     # ≈ 0.64
```

## ROC Curve

receiver operating characteristic (ROC) curve is another common tool used with binary classifiers
similar to precision/recall curve but plots `true positive rate (another name for recall) against the false positive rate`

`FPR` is the ratio of negative instances that are incorrectly classified as positive

$$
[
    fpr=1-tnr
]
$$

tnr(or specificity) is ratio of negative instances that are correctly classified as negative.

ROC curve plots sensitivity (recall) versus 1 – specificity.

you gotta compute TPR and FPR via:

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

to plot it:

```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
```

to compare this classifier we have a way: compare the area under curve
A perfect classifier will have a ROC AUC equal to 1, whereas
A purely random classifier will have a ROC AUC equal to 0.5

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
```

example:
train a RandomForestClassifier and compare its ROC curve and ROC AUC score to the SGDClassifier.

```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()
```

## Multiclass Classification

Unlike binary classification (only two possible outcomes), **multiclass classification** involves **three or more classes**.
Example: classifying digits 0–9 from the MNIST dataset.

Some algorithms (like `RandomForestClassifier`) handle this directly,
while others (like `SGDClassifier` or `SVC`) are binary and need special strategies.

### Strategies

#### One-vs-All (OvA)

Creates **one binary classifier per class**.
Each classifier decides “is this class or not?”
The class with the **highest confidence score** wins.

#### One-vs-One (OvO)

Creates a **binary classifier for every pair of classes**.
For `n` classes → `n × (n − 1) / 2` classifiers.
Each classifier votes → the class with the **most votes** is predicted.

```python
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
len(ovo_clf.estimators_)  # 45 classifiers for 10 digits
```

`RandomForestClassifier` can handle multiclass directly:

```python
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
forest_clf.predict_proba([some_digit])
# [[0.1, 0., 0., 0.1, 0., 0.8, 0., 0., 0., 0.]]
```

---

### Evaluating Multiclass Classifiers

Use **cross-validation** to check accuracy:

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```

Then compute a **confusion matrix** to visualize errors:

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```

Bright squares **off the diagonal** → misclassifications
Use this to analyze which digits (or classes) are being confused most often.

---

## Multilabel Classification

In **multilabel classification**, each instance can have **multiple labels** simultaneously.
Example: a face-recognition system might tag a photo as both “Alice” and “Bob”.

### Example

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)  # large digit
y_train_odd = (y_train % 2 == 1)  # odd digit
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
# [[False, True]] → Not large, but odd
```

### Evaluating Multilabel Models

Use **F1 score** averaged across labels:

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```

- `average="macro"` → unweighted average
- `average="weighted"` → accounts for label frequency

---

## Multioutput Classification

**Multioutput (or multioutput–multiclass)** generalizes multilabel classification:
each label can have **more than two possible values**.

Example: **image denoising** — predicting the clean version of a noisy image.

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
y_train_mod = X_train  # target = clean image

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```

Here, the model predicts a **vector of 784 pixel values** for each input image —
essentially performing **multioutput regression/classification**.

## Summary

| Type            | Description                             | Example                  |
| --------------- | --------------------------------------- | ------------------------ |
| **Binary**      | One of two classes                      | Spam vs Not Spam         |
| **Multiclass**  | One of many classes                     | Digit 0–9                |
| **Multilabel**  | Multiple binary labels per instance     | Detecting multiple faces |
| **Multioutput** | Multiple multiclass labels per instance | Image denoising          |