import pandas as pd
import numpy as np
from utils.df_adjustments import add_matrix_colour

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