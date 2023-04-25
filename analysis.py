import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import math
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score)


gk_data = pd.read_csv('final_data/goalkeeper_data.csv')
fp_data = pd.read_csv('final_data/field_player_data.csv')

gk_df = gk_data[gk_data['GP']>12]
fp_data = fp_data[fp_data['GP']>12]
fp_data['SC'] = fp_data['SC%']

gk_be_df = gk_df[gk_df['best_11'] == 1]

mid_df = fp_data[fp_data['POS'] =='M']
mid_be_df = mid_df[mid_df['best_11'] == 1]

fw_df = fp_data[fp_data['POS'] =='F']
fw_be_df = fw_df[fw_df['best_11'] == 1]

def_df = fp_data[fp_data['POS'] =='D']
def_be_df = def_df[def_df['best_11'] == 1]

important_cols = ['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC', 'RC', 'W', 'L', 'D']
#comparison of all players compared to best 11 at each position

#forwards
sns.boxplot(data = fw_df[important_cols])
sns.boxplot(data = fw_be_df[important_cols])
#midfielders
sns.boxplot(data = mid_df[important_cols])
sns.boxplot(data = mid_be_df[important_cols])
#defense
sns.boxplot(data = def_df[important_cols])
sns.boxplot(data = def_be_df[important_cols])


gk_important_cols = ['GP', 'SHTS', 'SV', 'GA', 'GAA', 'W', 'L', 'T', 'Sv%']

sns.boxplot(data = gk_df[gk_important_cols])
sns.boxplot(data = gk_be_df[gk_important_cols])


#Logistic Regression 
fw_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC', 'RC', 'W', 'L', 'D', 'best_11']].corr()
mid_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC', 'RC', 'W', 'L', 'D', 'best_11']].corr()
def_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC', 'RC', 'W', 'L', 'D', 'best_11']].corr()
gk_df[['GP', 'SHTS', 'SV', 'GA', 'GAA', 'W', 'L', 'T', 'Sv%', 'best_11']].corr()

fw_df= fw_df.dropna(subset=['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC',  'W', 'L'])
mid_df= mid_df.dropna(subset=['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC',  'W', 'L'])
def_df= def_df.dropna(subset=['GP', 'G',  'SHTS', 'SOG',  'W', 'L'])


fw_model = smf.logit("best_11 ~ GP + G + A + SHTS + SOG + SC + YC + W + L", data = fw_df).fit()
fw_model.summary()
fw_y_predict = fw_model.predict(fw_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC',  'W', 'L']])
#make binary zero/one from the percentages
fw_prediction = list(map(round, fw_y_predict))
# create a confustion matrix
fw_cm = confusion_matrix(fw_df[['best_11']], fw_prediction)
print ("Confusion Matrix : \n", fw_cm)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
fw_cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
fw_cm.flatten()/np.sum(fw_cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(fw_cm, annot=labels, fmt='', cmap='Blues')
plt.show()



mid_model = smf.logit("best_11 ~ GP + G + A + SHTS + SOG + SC + YC + W + L", data = fw_df).fit()
mid_model.summary()
mid_y_predict = mid_model.predict(mid_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC',  'W', 'L']])
#make binary zero/one from the percentages
mid_prediction = list(map(round, mid_y_predict))
# create a confustion matrix
mid_cm = confusion_matrix(mid_df[['best_11']], mid_prediction)
print ("Confusion Matrix : \n", mid_cm)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
mid_cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
mid_cm.flatten()/np.sum(mid_cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(mid_cm, annot=labels, fmt='', cmap='Blues')
plt.show()



def_model = smf.logit("best_11 ~ GP + G + SHTS + SOG + W + L", data = fw_df).fit()
def_model.summary()
def_y_predict = def_model.predict(def_df[['GP', 'G', 'A', 'SHTS', 'SOG', 'SC', 'YC',  'W', 'L']])
#make binary zero/one from the percentages
def_prediction = list(map(round, def_y_predict))
# create a confustion matrix
def_cm = confusion_matrix(def_df[['best_11']], def_prediction)
print ("Confusion Matrix : \n", def_cm)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
def_cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
def_cm.flatten()/np.sum(def_cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(def_cm, annot=labels, fmt='', cmap='Blues')
plt.show()

