import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

#https://www.kaggle.com/datasets/drgilermo/nba-players-stats
#df = pd.read_csv("stats_basic.csv")#.sample(1000) #.sample for better viewing of groups, less people to read through
df = pd.read_csv("stats_all.csv")
names = df["Player"]
years = df["Year"]

# FG stats, 2p, 3p and FT
# Rebounding, Asts, Steals, Blocks, turnovers, personal fouls and points
# Total for the season, not per game stats
X = df.iloc[:, 6:-1].to_numpy()

# normalizing it because points and shot attempts will overshadow stats
# like blocks and steals which do not get very high past 100 on some of the best players
std = np.std(X, axis=0)
X /= std


#########################
####DBScan Code##########
# dbv.plot_elbow((X))
#
# dbv.dbscan_cluster(X, min_points=42, epsilon=4)

# dbscan = DBSCAN(eps= 4, min_samples = 42).fit(X)

########################################
#######Find number of clusters#########
# inertia = np.zeros(10)
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, n_init=50).fit(X)
#     inertia[i-1] = kmeans.inertia_
#
# plt.plot(np.arange(1, 11), inertia)
# plt.show()


kmeans = KMeans(n_clusters = 4, n_init=50)
kmeans.fit(X)

# print(kmeans.inertia_)


for i in range(4):
    print(kmeans.cluster_centers_[i] * std)
    for name in names[kmeans.labels_ == i]:
        print(name)

########################################################################

###########Analysis on basic stats clustering###########################
# 4 clusters ended up clustering the players into different sort of categories that I could pick apart traits that were similar.
# 1 category would have players from a variety of positions but the biggest similarity to stand out were that they were consistent role players. Players who played
# a ton of minutes and games but were not the star players of those teams.
# Another category clearly had like a ton of centers who played down low in the post and grabbed rebounds but then some small guards
# would also be in that category like Russell Westbrook and Dwayne Wade because of their great rebounding even at a small size.
# I predicted that the players would likely be split up with more alignment to their positions but since the amount of clusters that fit well were 4,
# the players were clustered more into their play-style. With certain players being placed in different groups depending on the season they head and where
# their skills were required the most for their team. This kind of clustering correlates more towards how basketball is played nowadays because
# to be a good basketball player, most of the time you need to excel in many things but especially one that your team requires from you. I have seen
# players go from only a scorer who played not much defense to being traded to a different team and immediately being their best defender
# because that is what the team required from them instead of their scoring.

#########################Analysis on all stats (basic and advanced)#####################
# The clustering ends up looking about the same with the advance stats leaning a little more into separating what kind of role that player has
# on the team like star player, second or third option, starter, bench player.

#########################Improvements########################################
# I want to be able to print the year alongside the name of the player to get the exact year because for early years or later years of hall of fame players
# may end up in categories you would not expect. This will clear up confusion if a name appears
# next to others that really shouldn't be there if you know basketball well.
