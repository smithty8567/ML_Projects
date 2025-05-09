import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#https://www.kaggle.com/datasets/nikhil7280/weather-type-classification?select=weather_classification_data.csv
df = pd.read_csv("weather_predict.csv")

# Select Labels, weather types
y = df.iloc[:, -1].copy().to_numpy()

#Features: Temp, humidity, wind speed, precipitation %, atmospheric pressure, uv index, visibility (km)
#Could not use cloud cover, season, and location.
#Season and location would be very helpful to distinguish raining versus snowing and raining versus cloudy.
#Some locations and seasons depend a lot on if rain would happen there.
X = df.iloc[:, 1:-1].copy().to_numpy()

X_train, y_train = X, y

#finding optimal max depth
#------------------------------------------------
rdt = RandomForestClassifier(oob_score=True)
parameters = {"max_depth" : range(2,10)}
grid_search = GridSearchCV(rdt, param_grid=parameters, cv=5)
grid_search.fit(X_train,y_train)
max_depth = grid_search.best_params_["max_depth"]
# print("Best max depth is {}".format(max_depth))

rclf = RandomForestClassifier(oob_score=True, max_depth= max_depth)
rclf.fit(X_train, y_train)

#finding optimal C in linearSVC
#-----------------------------------------------------
dt = LinearSVC()
parameters = {"C" : np.linspace(1,100,num = 10)}
grid_search = GridSearchCV(dt, param_grid=parameters, cv=5)
grid_search.fit(X_train,y_train)
c = grid_search.best_params_["C"]
#print("Best C is {}".format(c))
# score_df = pd.DataFrame(grid_search.cv_results_)
# print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])

clf = LinearSVC(C=c)
bag_clf = BaggingClassifier(clf,n_estimators=100, oob_score=True, n_jobs = -1)
bag_clf.fit(X_train, y_train)


print(f"Random Forest Score (Train): {rclf.score(X_train, y_train):.3f}")
#print(f"Random Forest Score (Test): {rclf.score(X_test, y_test):.3f}")
print(f"Random Forest OOB Score: {rclf.oob_score_:.3f}")

print(f"SVC Score (Train): {bag_clf.score(X_train, y_train):.3f}")
#print(f"SVC Score (Test): {bag_clf.score(X_test, y_test):.3f}")
print(f"SVC OOB Score: {bag_clf.oob_score_:.3f}")

cm = confusion_matrix(y_train, rclf.predict(X_train))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=rclf.classes_)
disp_cm.plot()

cm = confusion_matrix(y_train, bag_clf.predict(X_train))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=bag_clf.classes_)
disp_cm.plot()

print(df.columns[1:-1])

importances = pd.DataFrame(rclf.feature_importances_, index=df.columns[1:-1])
importances.plot.bar()
plt.show()

