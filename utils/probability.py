import pandas as pd
import numpy as np
from scipy.stats import (
    binom, beta, poisson, expon, geom, nbinom,
    chi2_contingency
)

class ProbabilityModelling:
  def __init__(self, df: pd.DataFrame,  outcome: str, n_trials: int = 19):
    self.df = df
    self.outcome = outcome
    self.nTrials = n_trials
    self.x = np.arange(0,self.nTrials+1)

    # Explicit attribute initialization
    self.pTbl: pd.DataFrame | None = None
    self.pH: float = 0.0
    self.dist: np.ndarray | None = None
    self.bayes_dist: np.ndarray | None = None
    self.cmf: np.ndarray | None = None
    self.mean: float = 0.0
    self.var: float = 0.0
    self.std: float = 0.0
    self.n: int = 0

  def data_prep(self, method_of_prep : str = "frequentist") -> None:

    # check which method to use count or proportions
    if method_of_prep == "frequentist":
      proportion_or_count = True
      col = 'proportion'
    elif method_of_prep == "bayesian":
      proportion_or_count = False
      col = 'count'
    else:
      raise ValueError("Invalid method_of_prep. Please choose 'frequentist' or 'bayesian'.")
      # calculate the proportions
    self.pTbl = self.df['FT_Result'].value_counts(normalize=proportion_or_count).reset_index()

    # take sample size
    self.n = self.pTbl[col].sum()
    # get target outcome
    self.pH = (
          self.pTbl
          .loc[self.pTbl['FT_Result']==self.outcome][col]
          .item()
          )

  def calculate_global_prior(self, all_data, team_status):
    """
    I want to calculate global prior for teams with little info - newly promoted
    this function will do just that by grouping and extracting the count of wins
    the newly promoted teams get.
    """
    if all_data is None or all_data.empty:
        raise ValueError("unfiltered_data/all_data must be provided for informative priors.")

    # use all the train dataset got global prior
    mask = all_data['HomeTeamLeagueStatus'] == team_status
    globalTbl = (
        all_data['FT_Result']
            .loc[mask]
            .value_counts()
            .reset_index()
         )

    globalAlpha = (
          globalTbl
          .loc[globalTbl['FT_Result']==self.outcome]['count']
          .item()
          )
    globalBeta = globalTbl['count'].sum() - globalAlpha

    self.globalPrior = [globalAlpha,globalBeta]

    return self.globalPrior


  def binomial_modelling(self) -> None:

    # create the distribution
    self.dist = binom.pmf(self.x,self.nTrials,self.pH)
    self.cmf = binom.cdf(self.x,self.nTrials,self.pH)
    # obtain distribution stdev and mean
    self.mean, self.var = binom.stats(self.nTrials, self.pH, moments='mv')
    self.std = np.sqrt(self.var)

  def geometric_modelling(self) -> None:

    # create the distribution
    self.dist = geom.pmf(self.x,self.pH)
    self.cmf = geom.cdf(self.x,self.pH)

    # obtain distribution stdev and mean
    self.mean, self.var = geom.stats(self.pH, moments='mv')
    self.std = np.sqrt(self.var)

  def beta_modelling(self, prior : str, unfiltered_data : pd.DataFrame = None, team_status:str = None) -> None:

    prior_arr = [1,1] if prior == 'uninformative' else self.calculate_global_prior(unfiltered_data,team_status)
    alpha = self.pH + prior_arr[0]
    beta_ = (self.n-self.pH) + prior_arr[1]
    # create the distribution
    self.bayes_dist = beta.pdf(np.linspace(0,1,500),alpha,beta_)

    # get the stats (update pH with beta mean)
    self.pH, self.varBetaDist = beta.stats(alpha,beta_,moments='mv')
    self.stdBetaDist = np.sqrt(self.varBetaDist)


  def show_distribution(self, bayesian_method : bool = False) -> None:
    fig, axes = plt.subplots(
        2,1,figsize=(10,6)
    )

    if not bayesian_method:
      # show results
      sns.barplot(
            x=self.x,
            y=self.dist,
            label=f'n={self.nTrials}, p={self.pH*100:.2f}, mu={self.mean:.0f}, stdev={self.std:.0f}',
            ax=axes[0]
      )

      sns.lineplot(
            x=self.x,
            y=self.cmf,
            ax=axes[1]
      )
    else:
      # show results
      sns.barplot(
            x=self.x,
            y=self.dist,
            label=f'n={self.nTrials}, p={self.pH*100:.2f}, mu={19*self.pH:.0f}, stdev={self.std:.0f}',
            ax=axes[0]
      )

      sns.lineplot(
            x=np.linspace(0,1,500),
            y=self.bayes_dist,
            label=f'n={self.nTrials}, p={self.pH*100:.2f}, mu={19*self.pH:.0f}, stdev={self.stdBetaDist*100:.3f}',
            ax=axes[1]
      )

    # Attach legend explicitly to both subplots
    axes[0].legend()
    axes[1].legend()

    # adjust the visual
    if "adjust_visuals" in globals():
      adjust_visuals(
          title="Probability Mass Function (19-Match Horizon)",
          x_label="Number of Outcomes",
          y_label="Probability",
          ax_obj=axes
      )
  def run_model(self, **kwargs):
    model_type = kwargs.get('model_type')
    prior = kwargs.get('prior')
    data_prep_method = kwargs.get('method_of_prep')
    team_status = kwargs.get('team_status')

    all_data = kwargs.get('all_data')


    # run data prep
    self.data_prep(method_of_prep=data_prep_method)
    if model_type=='binomial':
      self.binomial_modelling()
      self.show_distribution()
    elif model_type=='geometric':
      self.geometric_modelling()
      self.show_distribution()
    elif model_type=='beta-binomial':
      self.beta_modelling(prior=prior, unfiltered_data=all_data, team_status=team_status)
      self.binomial_modelling()
      self.show_distribution(bayesian_method=True)
    else:
      raise ValueError("Invalid model_type. Please choose 'binomial', 'geometric' or beta-binomial.")

