import matplotlib.pyplot as plt



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