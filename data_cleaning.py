import pandas as pd
import numpy as np
from unidecode import unidecode

#Data Cleaning for Best 11 Data
best_eleven = pd.read_csv("raw_data/MLS_Best_11_Teams.csv")
best_eleven['Year'] = best_eleven['Year'].str[:4]

best_eleven_long = best_eleven.melt(id_vars=['Year'], value_vars=['Goalkeeper', 'Defenders', 'Midfielders', 'Forwards'])
best_eleven_long['value'] = best_eleven_long['value'].apply(lambda x: x.replace(', ', ','))
best_eleven_long['value'] = best_eleven_long['value'].apply(lambda x: x.replace('\n ', ','))

df_list = []
for i in range(len(best_eleven_long)):
    position = best_eleven_long['variable'].values[i]
    year = best_eleven_long['Year'].values[i]
    player_string  = best_eleven_long['value'].values[i]
    num_players = int(len(best_eleven_long['value'].values[i].split(','))/2)
    player_dict = {
    'player_0' : np.nan,
    'player_1' : np.nan,
    'player_2' : np.nan,
    'player_3' : np.nan,
    'team_0' : np.nan,
    'team_1' : np.nan,
    'team_2' : np.nan,
    'team_3' : np.nan
    }
    for i in range(num_players):
        pair = player_string.split(',')[i*2:i*2+2]
        player_dict[f'player_{str(i)}'] = pair[0]
        player_dict[f'team_{str(i)}'] = pair[1]
    for key in [('player_0', 'team_0'), ('player_1', 'team_1'), ('player_2', 'team_2'), ('player_3', 'team_3')]:
        temp_df = pd.DataFrame({'year': year,
                                'position':position,
                                'player_name': player_dict[key[0]],
                                'team': player_dict[key[1]]}, index = [0])
        df_list.append(temp_df)

best_eleven_df = pd.concat(df_list)
best_eleven_df.dropna(inplace=True)

best_eleven_df['player_name'] = best_eleven_df['player_name'].apply(unidecode)

best_eleven_df.reset_index(inplace=True)
del(best_eleven_df['index'])
best_eleven_df.columns = ['Year', 'POS', 'Player', 'Club']

best_eleven_club_dict = {
    'Dallas': 'DAL',  'Columbus':'CLB', 'Chicago':'CHI', 'LA Galaxy':'LA',  'Kansas City':'KC', 'MetroStars':'MET',
 'San Jose':'SJ', 'Colorado':'COL', 'D.C. United':'DC', 'Chivas USA':'CHV', 'Seattle':'SEA', 'Sporting KC':'KC',
 'Portland':'POR', 'Red Bulls':'NY', 'Philadelphia':'PHI', 'Minnesota':'MIN', 'New England':'NE', 'Miami':'MIA',
 'Houston':'HOU', 'Salt Lake':'RSL', 'Montreal':'MTL', 'Vancouver':'VAN', 'Toronto':'TOR', 'Atlanta':'ATL', 'LAFC':'LAFC',
 'Nashville':'NSH', 'Tampa Bay':'TB', 'New York City':'NYC', 'Cincinnati':'CIN', 'Austin':'ATX'}

for i in range(len(best_eleven_df)):
    best_eleven_df['Club'].values[i] = best_eleven_club_dict[best_eleven_df['Club'].values[i]]
    best_eleven_df['POS'].values[i] = best_eleven_df['POS'].values[i][0]

abbreviation_dict = {
    'Tampa Bay Mutiny':'TB', 
    'D.C. United': 'DC', 
    'NY/NJ MetroStars': 'MET',
    'Columbus Crew': 'CLB', 
    'New England Revolution': "NE",
    'Los Angeles Galaxy': 'LA',
    'Dallas Burn': "DAL", 
    'Kansas City Wiz': "KC", 
    'San Jose Clash': 'SJ',
    'Colorado Rapids': "COL", 
    'Kansas City Wizards': "KC", 
    'MetroStars': 'MET',
    'Miami Fusion': 'MIA', 
    'Chicago Fire' : 'CHI', 
    'San Jose Earthquakes': 'SJ',
    'FC Dallas': 'DAL', 
    'Real Salt Lake': 'RSL', 
    'Chivas USA': 'CHV', 
    'New York Red Bulls': 'NY',
    'Houston Dynamo': 'HOU', 
    'LA Galaxy': 'LA', 
    'Toronto FC':  'TOR', 
    'Seattle Sounders FC': 'SEA',
    'Philadelphia Union':'PHI', 
    'Sporting Kansas City':'KC', 
    'Portland Timbers': 'POR',
    'Vancouver Whitecaps FC': 'VAN', 
    'Montreal Impact': 'MTL', 
    'Columbus Crew SC':'CLB',
    'Orlando City SC':'ORL', 
    'New York City FC': 'NYC', 
    'Atlanta United FC':'ATL',
    'Minnesota United FC':'MIN', 
    'Los Angeles FC':'LAFC', 
    'FC Cincinnati':'CIN',
    'Nashville SC':'NSH', 
    'Inter Miami CF': 'MIA', 
    'Chicago Fire FC': 'CHI', 
    'CF Montreal': 'MTL',
    'Houston Dynamo FC': "HOU", 
    'Austin FC':'ATX',
    'Charlotte':'CHA',
    'Sporting KC': 'KC',
    'Seattle':'SEA',
    'Vancouver':'VAN',
    'Minnesota Utd':'MIN',
    'San Jose': 'SJ',
    'Austin':'ATX',
    'New England': 'NE',
    'Philadelphia':'PHI',
    'Nashville':'NSH',
    'NYCFC':'NYC',
    'Atlanta Utd':'ATL',
    'Orlando City':'ORL',
    'NY Red Bulls':'NY',
    'CF Montréal': 'MTL',
    'Inter Miami':'MIA'
    }

two_one_tables = pd.concat([pd.read_csv('raw_data/west_table_21.csv'),pd.read_csv('raw_data/east_table_21.csv')])
two_two_tables = pd.concat([pd.read_csv('raw_data/west_table_22.csv'),pd.read_csv('raw_data/east_table_22.csv')])
two_one_tables['Year'] = 2021
two_two_tables['Year'] = 2022
other_year_tables = pd.concat([two_one_tables,two_two_tables])
other_year_tables['GD'] = other_year_tables['GF'] - other_year_tables['GA']
other_year_tables['Club'] = other_year_tables['Squad']
other_year_tables = other_year_tables[['Club','Year','W','L','D','GD']]
#Table Data Cleaning
table_data = pd.read_csv('raw_data/all_tables.csv')  
table_data = table_data[table_data['Year'] != 2021]  
del(table_data['SW'])
del(table_data['SL'])
del(table_data['Head-to-head'])
del(table_data['PPG'])
del(table_data['GD'])
table_data['D'] = table_data['D'].fillna(0)
table_data['D'] = table_data['D'].astype(int)
table_data['Pos'] = table_data['Pos'].astype(int)
table_data['GD'] = table_data['GF'] - table_data['GA']
table_data['Team'].unique()
name_probz = [' (X)', ' (C)', ' (C, X)', ' (SS)', 's - ', 'x - ', 'y - ', ' (SS, W1)', ' (C1)', 'y ‚Äì ', 'x ‚Äì ', ' (SS) (W1)', ' (W1)', ' (E2)', ' (E3)', ' (SS, E1)', ' (E1)', 's ‚Äì ', ' (W2)', ' (W3)', '[a]', '[b]', ' (U)', ' (V)', ' 1', ' (S)', ' (M)', '2', '[e]']



for i in range(len(table_data['Team'].values)):
    for prob in name_probz:
        if prob in table_data['Team'].values[i]:
          table_data['Team'].values[i] = table_data['Team'].values[i].replace(prob, '')
        else: pass
        if 'Montr√©al' in table_data['Team'].values[i]:
            table_data['Team'].values[i] = table_data['Team'].values[i].replace('Montr√©al', 'Montreal')
        else: pass


table_data = table_data[table_data['Conference'] == 'Overall']
table_data['Club'] = table_data['Team']
table_data = table_data[['Club','Year','W','L','D','GD']]
table_data = pd.concat([table_data,other_year_tables])

for i in range(len(table_data['Club'].values)):
    for key in abbreviation_dict.keys():
        if table_data['Club'].values[i] == key:
            table_data['Club'].values[i] = abbreviation_dict[key]
        else: pass





all_players = pd.read_csv('raw_data/all_players.csv')
for i in range(len(all_players)):
    if all_players['Club'].values[i] == 'SKC':
        all_players['Club'].values[i] = 'KC'
    else: pass
    if all_players['Club'].values[i] in ['RBNY', 'NYRB']:
        all_players['Club'].values[i] = 'NY'
    all_players['POS'].values[i] = all_players['POS'].values[i][0]

all_players = all_players[all_players['Season'] == 'reg']
all_players = all_players[['Player', 'POS', 'Club', 'GP', 'GS', 'MINS', 'G', 'A', 'YC', 'RC', 'SHTS', 'SOG', 'SOG%', 'SC%', 'Year']]

std_two_one = pd.read_csv('raw_data/std_stats_21.csv')
shooting_two_one = pd.read_csv('raw_data/shooting_stats_21.csv')
std_two_two = pd.read_csv('raw_data/std_stats_22.csv')
shooting_two_two = pd.read_csv('raw_data/shooting_stats_22.csv')

twenty_one_data = std_two_one.merge(shooting_two_one, how = 'left', on = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born'])
twenty_one_data['Year'] = 2021
twenty_two_data = std_two_two.merge(shooting_two_two, how = 'left', on = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born'])
twenty_two_data['Year'] = 2022
last_two_years = pd.concat([twenty_one_data, twenty_two_data])
last_two_years = last_two_years[['Player', 'Pos', 'Squad', 'MP', 'Starts', 'Min', 'Gls_x', 'Ast', 'CrdY', 'CrdR', 'Sh', 'SoT', 'SoT%', 'G/Sh', 'Year']]
last_two_years.columns = ['Player', 'POS', 'Club', 'GP', 'GS', 'MINS', 'G', 'A', 'YC', 'RC', 'SHTS', 'SOG', 'SOG%', 'SC%', 'Year']
last_two_years['SC%'] = last_two_years['SC%']*100

last_two_years['Club'].unique

last_two_years_dict = {
    'Philadelphia': 'PHI',  'Columbus Crew': 'CLB',  'San Jose':'SJ', 'D.C. United':'DC', 'Minnesota Utd':'MIN', 
 'Colorado Rapids':'COL', 'NYCFC':'NYC', 'Toronto FC':'TOR', 'FC Dallas':'DAL', 'LA Galaxy':'LA',
 'FC Cincinnati':'CIN', 'Atlanta Utd':'ATL', 'Seattle':'SEA', 'Orlando City':'ORL', 'Nashville':'NSH',
 'Vancouver':'VAN', 'Chicago Fire':'CHI', 'Inter Miami':'MIA', 'NY Red Bulls':'NY', 'Los Angeles FC':'LAFC',
 'Portland Timbers':'POR', 'Houston Dynamo':'HOU', 'Sporting KC':'KC', 'CF Montréal':'MTL',
 'New England':'NE', 'Austin':'ATX', 'Real Salt Lake':'RSL', 'Charlotte':'CHA'
}
for i in range(len(last_two_years)):
    last_two_years['Club'].values[i] = last_two_years_dict[last_two_years['Club'].values[i]]
    last_two_years['POS'].values[i] = last_two_years['POS'].values[i][0]

field_player_data = pd.concat([all_players,last_two_years])
field_player_data = field_player_data[field_player_data['POS'] != 'G']
field_player_data['Player'] = field_player_data['Player'].apply(unidecode)


gk_two_one = pd.read_csv('raw_data/gk_stats_21.csv')
gk_two_one['Year'] = 2021
gk_two_two = pd.read_csv('raw_data/gk_stats_22.csv')
gk_two_two['Year'] = 2022
gk_last_two = pd.concat([gk_two_one,gk_two_two])
gk_last_two = gk_last_two[['Player', 'Pos', 'Squad', 'MP', 'Starts', 'Min', 'GA', 'GA90', 'Saves', 'SoTA', 'Save%', 'W', 'L', 'D', 'Year']]
gk_last_two.columns = ['Player', 'POS', 'Club', 'GP', 'GS', 'MINS', 'GA', 'GAA', 'SV', 'SHTS', 'Sv%', 'W', 'L', 'T', 'Year']
all_gk = pd.read_csv('raw_data/all_goalkeepers.csv')
all_gk = all_gk[all_gk['Season'] == 'reg']

for i in range(len(gk_last_two)):
    gk_last_two['Club'].values[i] = last_two_years_dict[gk_last_two['Club'].values[i]]

goalkeeper_data = pd.concat([all_gk, gk_last_two])
goalkeeper_data['Player'] = goalkeeper_data['Player'].apply(unidecode)

for i in range(len(goalkeeper_data)):
    if goalkeeper_data['Club'].values[i] == 'SKC':
        goalkeeper_data['Club'].values[i] = 'KC'
    else: pass
    if goalkeeper_data['Club'].values[i] in ['RBNY', 'NYRB']:
        goalkeeper_data['Club'].values[i] = 'NY'
    goalkeeper_data['POS'].values[i] = goalkeeper_data['POS'].values[i][0]


field_player_all_star_list = []
for i in range(len(best_eleven_df)):
    if best_eleven_df['POS'].values[i] != 'G':
        field_player_all_star_list.append([best_eleven_df['Player'].values[i], best_eleven_df['Club'].values[i], int(best_eleven_df['Year'].values[i])])
    else: pass
gk_all_star_list = []
for i in range(len(best_eleven_df)):
    if best_eleven_df['POS'].values[i] == 'G':
        gk_all_star_list.append([best_eleven_df['Player'].values[i], best_eleven_df['Club'].values[i], int(best_eleven_df['Year'].values[i])])
    else: pass

field_player_data['best_11'] = None
goalkeeper_data['best_11'] = None

field_player_data.drop_duplicates(inplace=True)
goalkeeper_data.drop_duplicates(inplace=True)




for i in range(len(field_player_data)):
    if ' ' in field_player_data['Club'].values[i]:
        field_player_data['Club'].values[i] = field_player_data['Club'].values[i].replace(' ','')
    else: pass
    if field_player_data['Player'].values[i][-1] == ' ':
        field_player_data['Player'].values[i] = field_player_data['Player'].values[i][:-1]
    else: pass
    if [field_player_data['Player'].values[i], field_player_data['Club'].values[i], field_player_data['Year'].values[i]] in field_player_all_star_list:
        field_player_data['best_11'].values[i] = 1
        field_player_all_star_list.remove([field_player_data['Player'].values[i], field_player_data['Club'].values[i], field_player_data['Year'].values[i]])
    else:
        field_player_data['best_11'].values[i] = 0

for i in range(len(goalkeeper_data)):
    if ' ' in goalkeeper_data['Club'].values[i]:
        goalkeeper_data['Club'].values[i] = goalkeeper_data['Club'].values[i].replace(' ','')
    else: pass
    if goalkeeper_data['Player'].values[i][-1] == ' ':
        goalkeeper_data['Player'].values[i] = goalkeeper_data['Player'].values[i][:-1]
    if [goalkeeper_data['Player'].values[i], goalkeeper_data['Club'].values[i], goalkeeper_data['Year'].values[i]] in gk_all_star_list:
        goalkeeper_data['best_11'].values[i] = 1
        gk_all_star_list.remove([goalkeeper_data['Player'].values[i], goalkeeper_data['Club'].values[i], goalkeeper_data['Year'].values[i]])
    else:
        goalkeeper_data['best_11'].values[i] = 0
field_player_data = field_player_data.merge(table_data, how='left', on = ['Club', 'Year'])
        
goalkeeper_data.to_csv('final_data/goalkeeper_data.csv')
field_player_data.to_csv('final_data/field_player_data.csv')