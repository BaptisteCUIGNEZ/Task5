import pandas as pd
import numpy as np

dataset = pd.read_csv('SDN_traffic.csv')

print('\n\nhead :\r')
print(dataset.head())
print('\n\ninfo : \r')
print(dataset.info())
print('\n\ndescribe : \r')
print(dataset.describe())
print('\n\nduplicated : \r')
print(dataset.duplicated())

X = dataset[["forward_bps_var",
             "tp_src", "tp_dst", "nw_proto",
             "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
             "forward_piat_mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
             "forward_pps_var",     "forward_pl_q1",    "forward_pl_q3",
             "forward_piat_q1",     "forward_piat_q3",  "forward_pl_max", "forward_pl_min",
             "forward_piat_max",    "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max",     "forward_bps_min", "forward_duration", "forward_size_packets",
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
             "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
             "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var",
             "reverse_pl_q1",    "reverse_pl_q3", "reverse_piat_q1",     "reverse_piat_q3",  "reverse_pl_max", 
             "reverse_pl_min", "reverse_piat_max",    "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
             "reverse_bps_max",     "reverse_bps_min", "reverse_duration", "reverse_size_packets","reverse_size_bytes" ]]

X.loc[1877, 'forward_bps_var'] = float(11968065203349)
X.loc[1931, 'forward_bps_var'] = float(12880593804833)
X.loc[2070, 'forward_bps_var'] = float(9022747730895)
X.loc[2381, 'forward_bps_var'] = float(39987497172945)
X.loc[2562, 'forward_bps_var'] = float(663300742992)
X.loc[2567, 'forward_bps_var'] = float(37770223877794)
X.loc[2586, 'forward_bps_var'] = float(97227875083751)
X.loc[2754, 'forward_bps_var'] = float(18709751403737)
X.loc[2765, 'forward_bps_var'] = float(33969277035759)
X.loc[2904, 'forward_bps_var'] = float(39204786962856)
X.loc[3044, 'forward_bps_var'] = float(9169996063653)
X.loc[3349, 'forward_bps_var'] = float(37123283690575)
X.loc[3507, 'forward_bps_var'] = float(61019064590464)
X.loc[3610, 'forward_bps_var'] = float(46849620984072)
X.loc[3717, 'forward_bps_var'] = float(97158873841506)
X.loc[3845, 'forward_bps_var'] = float(11968065203349)
X.loc[3868, 'forward_bps_var'] = float(85874278395372)



X['forward_bps_var'] = pd.to_numeric(X['forward_bps_var'])

print(X.info())


Y = dataset[['category']]
Y = Y.to_numpy()
Y = Y.ravel()

labels, uniques = pd.factorize(Y)
Y = labels
Y = Y.ravel() 
print('New value code of the Category column (but reshaped in horizontal vector)')
print(Y)


import scipy.stats as stats
X = stats.zscore(X)
X = np.nan_to_num(X)

#init and training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=42, test_size=0.7)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
clf = DecisionTreeClassifier(random_state=0, max_depth=6)
clf.fit(X_train, Y_train)
print(clf)

#Evaluation
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy = clf.score(X_test, Y_test)
Kfold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Kfold10 accuracy mean : ")
print(Kfold10_accuracy.mean())
predict = clf.predict(X_test)
cm = confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average='weighted', labels=np.unique(predict))
recall = recall_score(Y_test, predict, average='weighted', labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average='macro', labels=np.unique(predict))
print("Classification report : ")
print(classification_report(Y_test, predict, target_names=uniques))


#importance of the features
importance = clf.feature_importances_
important_feature_dict = {}

for idx, val in enumerate(importance):
    important_feature_dict[idx] = val
important_feature_list = sorted(important_feature_dict, key=important_feature_dict.get, reverse=True)

print(f'The 10 most important features :  {important_feature_list[:10]}')



fn = ["forward_bps_var",
             "tp_src", "tp_dst", "nw_proto",
             "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
             "forward_piat_mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
             "forward_pps_var",     "forward_pl_q1",    "forward_pl_q3",
             "forward_piat_q1",     "forward_piat_q3",  "forward_pl_max", "forward_pl_min",
             "forward_piat_max",    "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max",     "forward_bps_min", "forward_duration", "forward_size_packets",
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
             "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
             "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var",
             "reverse_pl_q1",    "reverse_pl_q3", "reverse_piat_q1",     "reverse_piat_q3",  "reverse_pl_max", 
             "reverse_pl_min", "reverse_piat_max",    "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
             "reverse_bps_max",     "reverse_bps_min", "reverse_duration", "reverse_size_packets","reverse_size_bytes" ]
la = ['WWW', 'DNS', 'FTP', 'ICMP', 'P2P', 'VOIP']
plt.figure(2, dpi=300)
fig = tree.plot_tree(clf, filled=True, feature_names=fn, class_names=la)
plt.title("Decision Tree trained on all the features")
plt.show()
print("c'est ok")