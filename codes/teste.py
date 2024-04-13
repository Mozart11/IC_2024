from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2) # Iris virginica

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)

X_new = [[5.5, 1.7], [5.0, 1.5]]
predicts = svm_clf.predict(X_new)

Xa = X[:,0]
Xb = X[:,1]

plt.scatter(Xa, np.arange(150), c="red")
plt.scatter(Xb, np.arange(150), c="blue")
plt.plot(predicts, c="black")
plt.show()
print("F")