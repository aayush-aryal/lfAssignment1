import pandas as pd 
from bs4 import BeautifulSoup
import re
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=BASE_DIR/"jobs.xlsx"


def read_get_info(path:str):
    df=pd.read_excel(path)
    print("Shape:",df.shape)
    print("Columns:", df.columns)
    print("Missing:", df.isna().sum())


def clean_job_description(text:str):
    soup=BeautifulSoup(text,"html.parser")
    clean_text=soup.get_text().lower()
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def clean_df_from_path(path:str):
    df=pd.read_excel(path)
    if "Job Description" in df.columns:
        df["Job Description"]=df["Job Description"].apply(lambda x: clean_job_description(x))
    df=df.fillna("Unknown")

    return df 



if __name__=="__main__":
    read_get_info(str(DATA_PATH))
    df=clean_df_from_path("/Users/aayush-aryal/Documents/lfAssignment1/jobs.xlsx")
    print(df.head())