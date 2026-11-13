import matplotlib.pyplot as plt
import seaborn as sns


# create a function that adjust visual axis
def adjust_visuals(title:str, x_label:str, y_label:str ,ax_obj=None):
  # create reuable title and label adjsutments dict
  title_params = {
        'fontsize':18,
        'fontweight':'bold',
        'color':'grey'
    }

  axis_params = {
        'fontsize':12,
        'fontweight':'bold',
        'color':'grey'
    }

  # we may encounter the use of subplots.
  # the below controls for this
  if not ax_obj is None:
    adjust_visual_obj(ax_obj,title,x_label,y_label,title_params,axis_params)
  else:
    # adjust the labels
    plt.xlabel(
        x_label,
        **axis_params
        );
    plt.ylabel(
        y_label,
        **axis_params
        );

    plt.title(
        title,
        **title_params
        );

    plt.grid(
        linestyle='--',
        alpha=0.25
    )
  # outside loop
  plt.tight_layout()





def adjust_visual_obj(ax_obj,title,x_label,y_label,title_params,axis_params):
  plt.suptitle(
      title,
      **title_params
      );

  # loop thorugh object and adjust
  for ax in ax_obj:
    ax.set_xlabel(
        x_label,
        **axis_params
        );
    ax.set_ylabel(
        y_label,
        **axis_params
    )
    ax.grid(
        linestyle='--',
        alpha=0.25
    )


def show_the_relartionship(df,**kwargs):

  #extract argss
  var1 = kwargs.get('var1')
  var2 = kwargs.get('var2')
  result = kwargs.get('result')
  plot_kind = kwargs.get('kind')
  title = kwargs.get('title')
  x_label_ = kwargs.get('x_label')
  y_label_ = kwargs.get('y_label')

  # show the distribution and relationships
  corr_tb = df[[var1,var2]].corr()
  corr_tb = corr_tb.round(2)
  print(f"Goals For & Against Correlation: \n\n {corr_tb}\n\n")

  plt.figure(
      figsize=(10,8)
  );

  sns.jointplot(
      data=df,
      x=var1,
      y=var2,
      hue=result,
      kind=plot_kind
    );


  # format the plot
  adjust_visuals(
      title=title,
      x_label=x_label_,
      y_label=y_label_
  )

def viz_the_validation(results:pd.DataFrame):
  """
  Here I will show the results of the cross validation over each fold

  """

  fig, axes = plt.subplots(
      len(results.columns)-1,1,figsize=(12,4 * len(results.columns))
  )


  if len(results) == 1:
      axes = [axes]  # ensure iterable when only 1 metric
  scoring = results.columns[2:]
  for ax, metric in zip(axes, scoring):
        sns.lineplot(
            data=results,
            x='Fold',
            y=metric,
            hue='Model',
            marker='o',
            ax=ax
        )
        ax.set_title(
            f'{metric} across folds',
            fontsize=14,
            fontweight='bold',
            color='grey'
        )
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.legend(title='Model')
        ax.grid(
            linestyle='--',
            alpha=0.5
        )

  # add overview title
  plt.suptitle(
      "Cross Validation Results",
       fontsize=20,
       fontweight='bold',
       color='grey',
       x=0.5,
       ha='center',
       y=1.01  # lifts it above the subplots
    )
  plt.tight_layout()