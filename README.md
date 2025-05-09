# ML_Projects

This is a repository that showcases multiple Machine Learning projects I have completed. They include clustering, neural networks, bagging, and random forests.

## Clustering Project
The clustering project uses data on NBA players' stats from there individual seasons dating back to the 1980s. I used both simple basketball stats and all stats including advance statistics as the features. The amount and type of features did not change the output of the types of clusters that were forming. A more in depth analysis is written at the end of the python file describing the summary of what the algorithm decided the cluster the players into.

## Neural Network
The neural network is ran on the same dataset as the clustering project. The neural network's job is to classify the players on 3 positions. I simplified the positions even more to just PG, SF, and C because of how basketball becomes positionless in the more recent years. This then makes the network have a harder time to train regardless of how many features are given because the amount of data is just not possible with how similar players play regardless of their position.

![image](https://github.com/user-attachments/assets/32805cf7-1eac-4607-9029-eb00c1ce0f34)
**Confusion Matrix for Neural Network**


## Random Forests and Bagging
The last two projects predict weather types and air pollution quality using both a random forest classifier and a bagging classifier with Linear SVC.

### Features for Weather
The features I used in the weather classifier were temperature, humidity, wind speed, precipitation %, atmospheric pressure, uv index, visibility (km). I dropped out features like season and location which would be very helpful to distinguish raining versus snowing and raining versus cloudy.

### Features for Pollution
The features used for pollution were temperature, humidity %, fine particle levels, coarse particle levels, nitrogen dioxide levels,
sulfur dioxide levels, carbon monoxide levels, distance to industrial zone (km), population density. This classification had an easier time classifying use random forests opposed to bagging because of the nice differences in quality between dense populations versus nondense ones.

![image](https://github.com/user-attachments/assets/9b7bca86-8789-478b-ac58-4346f74e60bf)
**Confusion Matrix for Random Forest on Pollution Classification**




