"""
 Step 1: Filter the dataset for NBA regular season data
nba_regular_season = df[df['league'] == 'NBA']"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


file_path = '/Users/connerstarkey/Downloads/players_stats_by_season_full_details.csv'
df = pd.read_csv(file_path)

print("Columns in the dataset:")
print(df.columns)

print("First few rows of the dataset:")
print(df.head())


nba_regular_season = df[df['League'] == 'NBA']

player_season_counts = nba_regular_season['Player'].value_counts()
most_seasons_player = player_season_counts.idxmax()
print("Player with the most regular seasons played: " + most_seasons_player)


player_data = nba_regular_season[nba_regular_season['Player'] == most_seasons_player].copy()


player_data['three_point_accuracy'] = player_data['3PM'] / player_data['3PA']

 
player_data['Season'] = player_data['Season'].apply(lambda x: int(x.split(' - ')[0]))


X = player_data[['Season']].values  
y = player_data['three_point_accuracy'].values  


reg = LinearRegression()
reg.fit(X, y)


y_pred = reg.predict(X)


plt.scatter(X, y, color='blue', label='Actual Accuracy')
plt.plot(X, y_pred, color='red', label='Line of Best Fit')
plt.xlabel('Season')
plt.ylabel('Three-Point Accuracy')
plt.title('Three-Point Accuracy Over Seasons')
plt.legend()
plt.show()

average_pred_accuracy = np.mean(y_pred)
actual_avg_accuracy = player_data['three_point_accuracy'].mean()

print("Average three-point accuracy from fit line: " + str(average_pred_accuracy))
print("Actual average three-point accuracy: " + str(actual_avg_accuracy))


missing_years = [2002, 2015]
for year in missing_years:
    interpolated_value = reg.predict([[year]])[0]
    print("Estimated three-point accuracy for the " + str(year) + " season: " + str(interpolated_value))


fgm_mean = nba_regular_season['FGM'].mean()
fgm_variance = nba_regular_season['FGM'].var()
fgm_skew = stats.skew(nba_regular_season['FGM'])
fgm_kurtosis = stats.kurtosis(nba_regular_season['FGM'])

fga_mean = nba_regular_season['FGA'].mean()
fga_variance = nba_regular_season['FGA'].var()
fga_skew = stats.skew(nba_regular_season['FGA'])
fga_kurtosis = stats.kurtosis(nba_regular_season['FGA'])

print("FGM Mean: " + str(fgm_mean))
print("FGM Variance: " + str(fgm_variance))
print("FGM Skew: " + str(fgm_skew))
print("FGM Kurtosis: " + str(fgm_kurtosis))
print("FGA Mean: " + str(fga_mean))
print("FGA Variance: " + str(fga_variance))
print("FGA Skew: " + str(fga_skew))
print("FGA Kurtosis: " + str(fga_kurtosis))


t_stat, p_value = stats.ttest_rel(nba_regular_season['FGM'], nba_regular_season['FGA'])
print("Paired t-test for FGM and FGA:")
print("T-statistic: " + str(t_stat))
print("P-value: " + str(p_value))

t_stat_fgm, p_value_fgm = stats.ttest_1samp(nba_regular_season['FGM'], fgm_mean)
t_stat_fga, p_value_fga = stats.ttest_1samp(nba_regular_season['FGA'], fga_mean)

print("Individual t-test for FGM:")
print("T-statistic: " + str(t_stat_fgm))
print("P-value: " + str(p_value_fgm))

print("Individual t-test for FGA:")
print("T-statistic: " + str(t_stat_fga))
print("P-value: " + str(p_value_fga))
