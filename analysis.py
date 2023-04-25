import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


gk_data = pd.read_csv('final_data/goalkeeper_data.csv')
fp_data = pd.read_csv('final_data/field_player_data.csv')

gk_df = gk_data[gk_data['GP']>10]
fp_data = fp_data[fp_data['GP']>10]

gk_be_df = gk_df[gk_df['best_11'] == 1]

mid_df = fp_data[fp_data['POS'] =='M']
mid_be_df = mid_df[mid_df['best_11'] == 1]

fw_df = fp_data[fp_data['POS'] =='F']
fw_be_df = fw_df[fw_df['best_11'] == 1]

def_df = fp_data[fp_data['POS'] =='D']
def_be_df = def_df[def_df['best_11'] == 1]

important_cols = ['GP', 'G', 'A', 'SHTS', 'SOG', 'SC%', 'YC', 'RC', 'W', 'L', 'D']
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
