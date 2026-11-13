import pandas as pd
import numpy as np
from utils.df_adjustments import add_matrix_colour
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import viz_the_validation

def calculate_win_pct(df: pd.DataFrame,team: str) -> pd.DataFrame:
  df = df.copy()
  sort_by_arg = 'H' if team == 'HomeTeam' else 'A'

  # run the pivot
  Wins = (
    df.pivot_table(
    index=team,
    columns='FT_Result',
    values='Season',
    aggfunc='count'
  )
  .sort_values(by=sort_by_arg,ascending=False)
  .reset_index()
  )

  # calculate the proportions
  Wins['H_pct'] = Wins['H']/Wins.select_dtypes(exclude='object').sum(axis=1)
  Wins['D_pct'] = Wins['D']/Wins.select_dtypes(exclude='object').sum(axis=1)
  Wins['A_pct'] = Wins['A']/Wins.select_dtypes(exclude='object').sum(axis=1)

  # drop counts
  Wins.drop(columns=['H','D','A'],inplace=True)

  # Show the matrix
  return add_matrix_colour(Wins,title=f"{team} Win Proportions")


def conduct_chi2(ct_observed, confidence):
  """
  This function is to simply re-use chi2 test as we are dealing
  with numerous counts for each feature.
  """
  # calculate the stats
  chi2, p, DoF, expected_value = chi2_contingency(ct_observed)

  # use if function to test whether the p surpasses confidence
  # to allow for rejecting of Null
  if p <= confidence:
    print("Reject Null Hypothesis -> There is evidence that the features are dependent")
  else:
    print("Fail to Reject Null Hypothesis")

  # print the stats
  print("*"*50)
  print(f"Chi2 Value: {chi2:.2f}")
  print(f"P-Value: {p:.2f}")
  print(f"Degrees of Freedom: {DoF}")
  print(f"Expected Value: \n{expected_value}")
  print("*"*50)

def contingency_table_(col1,target,title=None,col2=None):
  # create contingency table
  if col2 is not None:
    ct = pd.crosstab(
        [col1,col2],
        target
    )
  else:
    ct = pd.crosstab(
        col1,
        target
    )

  # adjust the table using add_matrix_colour
  return add_matrix_colour(ct,title=title)

def run_chi2_test(**kwargs):
  #extract the inputs
  var1 = kwargs.get('var1')
  target = kwargs.get('target')
  confidence = kwargs.get('confidence')

  home_goals_ct = pd.crosstab(
        var1,
        target
    )
  # show the table
  tbl = contingency_table_(
        var1,
        target=target
    )

  # perform chi2 test
  conduct_chi2(home_goals_ct,confidence=confidence)



  return tbl


def goals_for_and_against_match_avg(df:pd.DataFrame, team:str, sortby:str) -> pd.DataFrame:
  #df = df.copy()

  # group by selected team
  grp = df.groupby(team)

  # agg the data
  goals_for_and_against = (
    grp[['FT_HomeGoals','FT_AwayGoals']]
    .mean()
    .reset_index()
    .sort_values(by=sortby,ascending=False)
  )

  return goals_for_and_against

def run_cross_validation(X_train_in, y_train_in, models, cv=5, scoring=None):
    """

    1. always resorts to accuracy if there is no scoring inputted
    2. The standard crossvalidation is applied. We do not have too heavy class imabalance so cross_validate()
       was appropriate and no stratification was selected.
    3. Create a new df that allows for the visualisation of mt results over fold.
    4. Visualise each result over the selected scoring.

    """
    if scoring is None:
        scoring = ['accuracy']

    results = []
    plot_data = []  # separate list for per-fold plot data

    # create timeseries crossvalisation
    tcsv = TimeSeriesSplit(n_splits=5)

    # create list to save results
    results = []

    for fold, (train_idx, val_idx) in enumerate(tcsv.split(X_train_in), start=1):
      # spplit the data by time chunks
      X_train_,X_val_ = X_train_in.iloc[train_idx],X_train_in.iloc[val_idx]
      y_train_,y_val_ = y_train_in[train_idx],y_train_in[val_idx]

      # loop through the models to make their predictions
      for model_name,model in models.items():
        model.fit(X_train_,y_train_)
        y_pred = model.predict(X_val_)

        #get f1
        f1_out = f1_score(y_val_,y_pred,average='macro')
        prec = precision_score(y_val_,y_pred,average='macro')
        recall = recall_score(y_val_,y_pred,average='macro')
        accuracy = accuracy_score(y_val_,y_pred)


        # save predictions and actuals as rows
        results.append({
                    "Fold": fold,
                    "Model": model_name,
                    "F1": f1_out,
                    "Accuracy": accuracy,
                    "Precision": prec,
                    "Recall": recall
                })
    # save to df
    results = pd.DataFrame(results)

    # show results in viz
    viz_the_validation(results)


