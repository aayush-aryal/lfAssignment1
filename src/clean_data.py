import pandas as pd 
from bs4 import BeautifulSoup



def read_get_info(path:str):
    df=pd.read_excel(path)
    print("Shape:",df.shape)
    print("Columns:", df.columns)
    print("Missing:", df.isna().sum())


#what does the data look like
read_get_info("/Users/aayush-aryal/Documents/lfAssignment1/jobs.xlsx")


def clean_job_description(text:str):
    soup=BeautifulSoup(text,"html.parser")
    return soup.get_text()


def clean_df_from_path(path:str):
    df=pd.read_excel(path)
    if "Job Description" in df.columns:
        df["Job Description"]=df["Job Description"].apply(lambda x: clean_job_description(x))
    if "Tags" in df.columns:
        df["Tags"]=df["Tags"].fillna("Unknown")
    return df 


df=clean_df_from_path("/Users/aayush-aryal/Documents/lfAssignment1/jobs.xlsx")
print(df.head())