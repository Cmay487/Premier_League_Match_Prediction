import pandas as pd
import numpy as np
from utils.df_adjustments import add_matrix_colour
from scipy.stats import chi2_contingency

def calculate_win_pct(df: pd.DataFrame,team: str) -> pd.DataFrame:
  df = df.copy()
  sort_by_arg = 'H' if team == 'HomeTeam' else 'A'

  # run the pivot
  Wins = (
    df.pivot_table(
    index=team,
    columns='FT_Result',
    values='League Division',
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