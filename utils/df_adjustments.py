import pandas as pd
import seaborn as sns

def add_matrix_colour(df,title=None):
  cm = sns.light_palette("green", as_cmap=True)
  styled = (
      df.style
      .background_gradient(cmap=cm)
      .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#1B2A4A'),
                ('color', 'white'),
                ('font-weight', 'bold')
            ]}
        ])
  )
  if title:
    styled = styled.set_caption(title)
  return styled