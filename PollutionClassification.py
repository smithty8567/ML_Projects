import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment
df = pd.read_csv("updated_pollution_dataset.csv")
df = df.dropna()

# Select variables
# Pollution quality
y = df.iloc[:, -1].copy().to_numpy()

# columns Temperature, Humidity %, fine particle levels, coarse particle levels, nitrogen dioxide levels,
# sulfur dioxide levels, carbon monoxide levels, distance to industrial zone (km), population density
X = df.iloc[:, :-1].copy().to_numpy()
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Train/test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rdt = RandomForestClassifier(oob_score=True)
parameters = {"max_depth" : range(2,10)}
grid_search = GridSearchCV(rdt, param_grid=parameters, cv=5)
grid_search.fit(X_train,y_train)
max_depth = grid_search.best_params_["max_depth"]
# print("Best max depth is {}".format(max_depth))

rclf = RandomForestClassifier(oob_score=True, max_depth= max_depth)
rclf.fit(X_train, y_train)

#score_df = pd.DataFrame(grid_search.cv_results_)
#print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

dt = LinearSVC()
parameters = {"C" : np.linspace(1,100,num = 10)}
grid_search = GridSearchCV(dt, param_grid=parameters, cv=5)
grid_search.fit(X_train,y_train)
c = grid_search.best_params_["C"]
# print("Best C is {}".format(c))

# score_df = pd.DataFrame(grid_search.cv_results_)
# print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])

clf = LinearSVC(C=c)
bag_clf = BaggingClassifier(clf,n_estimators=10, oob_score=True, n_jobs = -1)
bag_clf.fit(X_train, y_train)

print(f"Random Forest Score (Train): {rclf.score(X_train, y_train):.3f}")
print(f"Random Forest Score (Test): {rclf.score(X_test, y_test):.3f}")
print(f"Random Forest OOB Score: {rclf.oob_score_:.3f}")

print(f"SVC Score (Train): {bag_clf.score(X_train, y_train):.3f}")
print(f"SVC Score (Test): {bag_clf.score(X_test, y_test):.3f}")
print(f"SVC OOB Score: {bag_clf.oob_score_:.3f}")

cm = confusion_matrix(y_test, rclf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=rclf.classes_)
disp_cm.plot()

cm = confusion_matrix(y_test, bag_clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=bag_clf.classes_)
disp_cm.plot()


plt.show()