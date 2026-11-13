import pandas as pd



def lagged_shots_feature(df:pd.DataFrame, team:str, shotsOT:str) -> pd.DataFrame:

  df = df.copy()

  df[f'{team}5MatchShotsOTMean'] = df.groupby(team)[shotsOT].transform(
      lambda x: x.shift(1).rolling(5, min_periods=1).mean()
      ).fillna(0)

  # control for conceded shots OT.
  # we are asking - how many shots OT did the home team leak?
  if team == 'HomeTeam':
    concededTeam = 'AwayShotsonTarget'
  else:
    concededTeam = 'HomeShotsonTarget'

  df[f'{team}5MatchShotsOTConcededMean'] = df.groupby(team)[concededTeam].transform(
      lambda x: x.shift(1).rolling(5, min_periods=1).mean()
      ).fillna(0)


  # place the feature into a bucket
  def new_bucket(shots):
    if shots >= 6:
      return '6+'
    elif shots >= 3:
      return '3-5'
    else:
      return '0-2'

  df[f'{team}5MatchShotsOTMeanBucket'] = df[f'{team}5MatchShotsOTMean'].apply(new_bucket)
  df[f'{team}5MatchShotsOTConcededMeanBucket'] = df[f'{team}5MatchShotsOTConcededMean'].apply(new_bucket)

  return df


def lagged_shots_feature_nOT(df:pd.DataFrame, team:str, shotsOT:str) -> pd.DataFrame:

  df = df.copy()

  df[f'{team}5MatchShotsMean'] = df.groupby(team)[shotsOT].transform(
      lambda x: x.shift(1).rolling(5, min_periods=1).mean()
      ).fillna(0)

  # control for conceded shots OT.
  # we are asking - how many shots OT did the home team leak?
  if team == 'HomeTeam':
    concededTeam = 'AwayShots'
  else:
    concededTeam = 'HomeShots'

  df[f'{team}5MatchShotsConcededMean'] = df.groupby(team)[concededTeam].transform(
      lambda x: x.shift(1).rolling(5, min_periods=1).mean()
      ).fillna(0)

  def new_bucket_shots(shots):
    if shots >= 15:
        return '15+'
    elif shots >= 8:
        return '8-14'
    else:
        return '0-7'

  df[f'{team}5MatchShotsMeanBucket'] = df[f'{team}5MatchShotsMean'].apply(new_bucket_shots)
  df[f'{team}5MatchShotsConcededMeanBucket'] = df[f'{team}5MatchShotsConcededMean'].apply(new_bucket_shots)

  return df



def rolling_goals_for_and_against(df:pd.DataFrame):
  df = df.copy()
  # ===============
  # last game
  # ===============

  # home team
  df['HomeTeamLastGameGoalsConceded'] = df.groupby('HomeTeam')['FT_AwayGoals'].transform(
      lambda x: x.shift(1)
  ).fillna(0)

  df['HomeTeamLastGameGoalsScored'] = df.groupby('HomeTeam')['FT_HomeGoals'].transform(
      lambda x: x.shift(1)
  ).fillna(0)


  #away team
  df['AwayTeamLastGameGoalsConceded'] = df.groupby('AwayTeam')['FT_HomeGoals'].transform(
      lambda x: x.shift(1)
  ).fillna(0)

  df['AwayTeamLastGameGoalsScored'] = df.groupby('AwayTeam')['FT_AwayGoals'].transform(
      lambda x: x.shift(1)
  ).fillna(0)
  # ===============
  # last 3-5 games
  # ===============
  df = rolling_goals_for_last_n_games(df)

  #return result
  return df


def rolling_goals_for_last_n_games(df:pd.DataFrame):

  for i in [3,5]:
    # home team
    df[f'HomeTeamLast{i}GamesGoalsConcededAvg'] = df.groupby('HomeTeam')['FT_AwayGoals'].transform(
        lambda x: x.shift(1).rolling(window=i,min_periods=1).mean()
    ).fillna(0)

    df[f'HomeTeamLast{i}GamesGoalsScoredAvg'] = df.groupby('HomeTeam')['FT_HomeGoals'].transform(
        lambda x: x.shift(1).rolling(window=i,min_periods=1).mean()
    ).fillna(0)

    #away team
    df[f'AwayTeamLast{i}GamesGoalsConcededAvg'] = df.groupby('AwayTeam')['FT_HomeGoals'].transform(
        lambda x: x.shift(1).rolling(window=i,min_periods=1).mean()
    ).fillna(0)

    df[f'AwayTeamLast{i}GamesGoalsScoredAvg'] = df.groupby('AwayTeam')['FT_AwayGoals'].transform(
        lambda x: x.shift(1).rolling(window=i,min_periods=1).mean()
    ).fillna(0)

  return df


def shot_acc_feature(df):
  df = df.copy()

  # create the features for lagged 5 games
  df = lagged_shots_feature_nOT(df,'HomeTeam','HomeShots')
  df = lagged_shots_feature_nOT(df,'AwayTeam','AwayShots')

  df['HomeTeam5MatchMeanShotAccConceded'] = (df['HomeTeam5MatchShotsOTConcededMean']/df['HomeTeam5MatchShotsConcededMean']).fillna(0)
  df['AwayTeam5MatchMeanShotAccConceded'] = (df['AwayTeam5MatchShotsOTConcededMean']/df['AwayTeam5MatchShotsConcededMean']).fillna(0)

  return df



def rolling_win_pct(df:pd.DataFrame, team:str) -> pd.DataFrame:
  df = df.copy()

  # sort df in asc order
  df.sort_index(inplace=True)

  # create helper col that detrmines home wins
  df['HomeWin'] = df.apply(lambda x: 1 if x['FT_Result'] == 'H' else 0, axis=1)
  df['AwayWin'] = df.apply(lambda x: 1 if x['FT_Result'] == 'A' else 0, axis=1)
  df['Draw'] = df.apply(lambda x: 1 if x['FT_Result'] == 'D' else 0, axis=1)

  # loop through the outcomes to calculate the roling win,loss and draw pct by team.
  outcome_list = ['HomeWin','AwayWin','Draw']
  for outcome in outcome_list:
    df[f'{team}{outcome}Pct'] = (
        df.groupby(team)[outcome].transform(
            lambda x: x.shift(1).expanding().sum()
            )/
        df.groupby(team)[outcome].transform(
            lambda x: x.shift(1).expanding().count()
            )
    ).fillna(0)

  # drop the helper columns
  df.drop(columns=['HomeWin','AwayWin','Draw'],inplace=True)

  return df


def adjust_season(season_str):
    # convert to int for easier reduction
    start_num = int(season_str[:2])
    end_num = int(season_str[2:])

    # subtract one year
    start_num -= 1
    end_num -= 1

    # convert back to 2-digit strings with leading zeros
    start_str = str(start_num).zfill(2)
    end_str = str(end_num).zfill(2)

    # combine into season string
    return start_str + end_str

def team_status(df:pd.DataFrame) -> pd.DataFrame:
  df = df.copy()

  # create a dict containing each team by season
  season_team_map = df.groupby("Season")["HomeTeam"].apply(set).to_dict()
  season_away_team_map = df.groupby("Season")["AwayTeam"].apply(set).to_dict()

  # helper col
  df["PreviousSeason"] = df["Season"].apply(adjust_season)

  # create the features
  df["HomeTeamLeagueStatus"] = df.apply(
      lambda row: "Newly Promoted"
                  if row['Season'] != '1819' and row["HomeTeam"] not in season_team_map.get(row["PreviousSeason"] , set())
                  else "Established",
      axis=1
  )

  df["AwayTeamLeagueStatus"] = df.apply(
      lambda row: "Newly Promoted"
                  if  row['Season'] != '1819' and row["AwayTeam"] not in season_team_map.get(row["PreviousSeason"] , set())
                  else "Established",
      axis=1
  )

  # drop helper col
  df.drop(columns=['PreviousSeason'],inplace=True)

  return df