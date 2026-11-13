# data manipulation
import pandas as pd
import numpy as np

# web scraping
from bs4 import BeautifulSoup
import requests

# create a class function to extract the raw data.
class extractWebData:
  def __init__(self, url, seasons_to_extract):
    self.soup_obj = None
    self.df = None
    self.csv_links = []
    self.url = url
    self.seasons_to_extract  = seasons_to_extract

  def scrape_the_data(self):
    """
    The first part of the function will scrape all the data that are csv files.
    I will then append all of these files to a dict for easy look up
    """
    # get page data
    web_page = requests.get(self.url)

    # create soup object
    self.soup_obj = BeautifulSoup(web_page.content, 'html.parser')




  def file_loop(self):
    """
    In this fuinction I will loop through the files and only save the
    Prem data in the last specified amount of seasons
    """
    for a_tag in self.soup_obj.find_all('a', href=True):
        href = a_tag['href']

        if href.lower().endswith('.csv'): #checks that the link is a csv file
          tag_name = a_tag.get_text(strip=True)
          tag_link = requests.compat.urljoin(self.url, href)
          # only want prem data!
          if tag_name == 'Premier League':
            self.csv_links.append(
                tag_link # only require the url to csv file
            )


  def convert_to_df(self):
    """
    Loop thorugh each url and save to list, where each csv will be joined
    into 1 df ready for data analysis.
    """
    # create a dataframe with the datasets
    all_dfs = []
    for i in range(self.seasons_to_extract): # extract 6 seasons worth of data
      prem = pd.read_csv(self.csv_links[i])
      prem['Season'] = self.csv_links[i].split('/')[4] # get the season name from the csv string
      all_dfs.append(prem)

    # drop into df
    self.df = pd.concat(all_dfs)


  def drop_unwanted_columns(self):
    """
    The data contains columns that appear in most recent seasons but not in historic ones.
    Drop these columns as they're likely bettering related.
    """
    # calculate the number of blank rows in each column
    list_of_nan = self.df.isna().sum().sort_values(ascending=False).reset_index()
    list_of_nan.columns = ['ColumnName', 'CountOfBlanks']

    # ony keep the features with 0 blanks as these are the key features
    columns_to_drop = list_of_nan['ColumnName'].loc[list_of_nan['CountOfBlanks']>0].to_list()
    self.df.drop(columns=columns_to_drop,inplace=True)

  def rename_the_columns(self):
    """
    The column names are not named well for data analysis.
    I will rename them for ease of use.
    """
    rename_dict = {
      # Match stats
      'Div': 'League Division',
      'FTHG': 'FT_HomeGoals',
      'FTAG': 'FT_AwayGoals',
      'FTR': 'FT_Result',
      'HTHG': 'HT_HomeGoals',
      'HTAG': 'HT_AwayGoals',
      'HTR': 'HT_Result',
      'Referee': 'Match Referee',
      'HS': 'HomeShots',
      'AS': 'AwayShots',
      'HST': 'HomeShotsonTarget',
      'AST': 'AwayShotsonTarget',
      'HF': 'HomeFoulsCommitted',
      'AF': 'AwayFoulsCommitted',
      'HC': 'HomeCorners',
      'AC': 'AwayCorners',
      'HY': 'HomeYellowCards',
      'AY': 'AwayYellowCards',
      'HR': 'HomeRedCards',
      'AR': 'AwayRedCards'
    }
    self.df.rename(columns=rename_dict, inplace=True)

  def run_web_scrape(self):
    self.scrape_the_data()
    self.file_loop()
    self.convert_to_df()
    self.drop_unwanted_columns()
    self.rename_the_columns()
    return self.df

# call class as object
scaper = extractWebData('https://www.football-data.co.uk/englandm.php', 8)
df = scaper.run_web_scrape()

df.info()