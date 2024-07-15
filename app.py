import arxiv
import pandas as pd
import numpy as np
from dotenv import load_dotenv 
import os
import requests
import re
#from datetime import datetime, date
import streamlit as st
import openai
#import random

##UDFs 

def extract_authors(author_list):
    names = [pd.Series(author).str.extract(r"'(.*?)'")[0].dropna().tolist() for author in author_list]
    flat_names = [name for sublist in names for name in sublist]
    return flat_names

# Process the author names to remove duplicates and join them
def process_authors(names_list):
    # Remove duplicates while preserving order
    unique_names = []
    for name in names_list:
        if name not in unique_names:
            unique_names.append(name)
    return ', '.join(unique_names)


def abstract_to_pdf(url):
    # Remove the versioning at the end
    if url.endswith('v1'):
        url = url[:-2]
    # Replace '/abs/' with '/pdf/' and add '.pdf' at the end
    return url.replace('/abs/', '/pdf/') + '.pdf'


def extract_score(text):
    # Search for the pattern "Overall Score: number/number" or "Overall Score: number"
    match = re.search(r'Overall Score: (\d+)(?:/5)?', text)
    # If a match is found, return the first group (the number) as an integer
    if match:
        return int(match.group(1))
    # If no match is found, you could return NaN or a default value
    return float('nan')

def extract_interest_score(text):
    # Search for the pattern "Overall Score: number/number" or "Overall Score: number"
    match = re.search(r'Interest Score: (\d+)(?:/5)?', text)
    # If a match is found, return the first group (the number) as an integer
    if match:
        return int(match.group(1))
    # If no match is found, you could return NaN or a default value
    return float('nan')

def extract_reading_time(text):
    # Search for the pattern "Overall Score: number/number" or "Overall Score: number"
    match = re.search(r'Reading Time: (\d+)(?:/5)?', text)
    # If a match is found, return the first group (the number) as an integer
    if match:
        return int(match.group(1))
    # If no match is found, you could return NaN or a default value
    return float('nan')



## Cleaning and Extracting

load_dotenv()
 
openai.api_key = os.getenv('OPENAI_API_KEY')
chatpdf_key = os.getenv('chatpdf_key')

def load_articles(paper_id):
    dfList = []

    ## Running Seach through arxiv api and extracting results of search terms

    

                
    search = arxiv.Search(
       id_list=[paper_id]
    )
    papers = pd.DataFrame()
    title       = []
    author      = []
    summary     = []
    updated     = []
    link        = []
    
        




    for result in arxiv.Client().results(search):
        title.append(result.title)
        author.append(result.authors)
        summary.append(result.summary)
        updated.append(result.updated)
        link.append(result.entry_id)
          



        papers = pd.DataFrame( list(zip(title, author, summary,
                                        updated, link
                                        )),
                       columns =['title', 'author', 'summary','updated',
                                 'link'])


        papers['author'] = papers['author'].astype(str)

        names = papers['author'].str.extractall(r"'(.*?)'")

        papers['author'] = names.groupby(level=0)[0].apply(', '.join)

        dfList.append(papers)


    combined_papers = pd.concat(dfList)

    combined_papers['pdf_url'] = combined_papers['link'].apply(abstract_to_pdf)

    return(combined_papers)
    
   

    




st.title("Summmariser")


st.sidebar.header("ArXiv Single Paper Summarizer")
st.sidebar.write("Enter paper Id to summarize for newsletter")

# Multi-select for categories

paper_id = st.sidebar.text_input("paper id", value="eg 1706.03762")

# Button to run the app
run_button = st.sidebar.button('Retrieve Arxiv Results')
# Initialize session state for the DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
# Display a loading message while processing
if run_button:
    with st.spinner('Fetching and processing data...'):

        st.session_state.df = load_articles(paper_id)

        headers = {
              'x-api-key': chatpdf_key,
              'Content-Type': 'application/json'
            }
        
        summary_dict = {}
        
        for index, row in st.session_state.df.iterrows():
                try:
                
                    data = {'url': row["pdf_url"]}
                    response = requests.post(
                        'https://api.chatpdf.com/v1/sources/add-url', headers=headers, json=data)
                    if response.status_code == 200:
                        print('Source ID:', response.json()['sourceId'])
                        source_id = response.json()['sourceId']
                    else:
                        print('Status:', response.status_code)
                        print('Error:', response.text)
                    qna_data = {
                        'sourceId': source_id,'messages': [
                            {
                                'role': "user",
                                'content': """You are a world class researcher, I want you to act as a research paper summarizer. 
                                                summarise the following paper.
                                                You will create a short summary of the main points and findings of the paper. You will follow the following rules as strictly as possible:
                                                1/ Your summary should be concise, easy to understand, and should accurately and objectively communicate 
                                                the key points of the paper.
                                                2/ You should not include any personal opinions or interpretations in your summary but rather focus on objectively presenting the 
                                                information from the paper. 
                                                3/ Your summary should accurately reflects the content of the original paper, do not make up anything.
                                              
                                                4/ Provide a approximate reading time based on the length of the paper, an example is "Reading Time: 10mins""",
                            }
                        ]
                    }   
                    response = requests.post(
                        'https://api.chatpdf.com/v1/chats/message', headers=headers, json=qna_data)
                    if response.status_code == 200:
                        result = response.json()['content']
                        print(f"success in row {index}")
                        print(result)
                        summary_dict[row['pdf_url']] = result
                    else:
                        print(f"failure in row {index}")
                        summary_dict[row['pdf_url']] = "failed"
                except:
                    pass    
                
            # Convert the dictionary to a DataFrame
        summary_df = pd.DataFrame(list(summary_dict.items()), columns=['pdf_url', 'result'])

        summary_df['reading_time'] = summary_df['result'].apply(extract_reading_time)

            # merging the df with summaries back onto the original df
        merged_df = summary_df.merge(st.session_state.df, on='pdf_url', how='left')
           
                       
         
            ## this file covers the history of files
            
            ## this file contains only the most current papers not previous covered
            #merged_df.to_csv("current-data/merged_current.csv", index = False)
            #quarantined.to_csv("quarantined/quarantined-papers.csv", index = False)
        columns_to_select = ["title", "author", "link", "result"]
        sub_df = merged_df[columns_to_select]

        for index, row in sub_df.iterrows():
                # Create a list of 'ColumnName: Value' for each column in the row
                row_str = ', '.join([f"{col}: {row[col]}" for col in sub_df.columns])

                with st.expander(f"Summmary", expanded=True):  
                    
                    response = openai.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                        {
                        "role": "system", #systems prompt
                        "content": """
                            You are a world class researcher, You will be given the details and summary of a research paper in AI, using your experience and knowledge 
                            as a researcher and academic:

                            1/ Give an overall  score of the paper based on how interesting, relevant, 
                            meaningful of a contribution and how original the papers is, you do not need to comment on the score other than to say "**AI Score:**" and
                            then giving a score out of 3, where a higher score represents a better quality paper, a score of 1 is an average paper, 
                             a score of 2 if a great paper and score of 3 is an excellent paper. instead of returning the  overall score as a number repeat the string ":rocket:" a number of times
                             corresponding to the score, for example overall score of 2 should be represented as "**AI Score**: :rocket: :rocket:".
                            
                            2/ Provide an "Interest Score" out of 3 based on how technical or easy to read the paper is, a score of 1 is very technical, 2 is accesible and 3 is very technical
                                instead of returning the  interest score as a number repeat the string ":magnet:" a number of times
                                corresponding to the score, for example an interest score of 2 should be represented as "**Interest Score**: :magnet:".
                            
                            3/ using the reading_time provided, output a reading time, if the reading time less than 10 mins  the it is "**Reading Time**: :alarm_clock:", 
                            if it is between 10mins to 20mins inclusive then "**Reading Time**: :alarm_clock: :alarm_clock:", if it is greater than 20 mins then "**Reading Time**: :alarm_clock: :alarm_clock: :alarm_clock:"

                            4/ Generate 5 key topics using your expertise as a researcher based on the summary of the paper given
                            
                            4/ Do not repeat the title of the paper again, make sure there is variability in the scoring

                            5/ Format it as the example below, do not deviate from this format, follow it strictly:


                            ## Multiple Instance Learning for Uplift Modeling

                            #### Yao Zhao, Haipeng Zhang, Shiwei Lyu

                            **Key Topics:** Uplift Modeling, Multiple Instance Learning, Marketing Campaigns, Treatment Effects, Algorithm Development

                            Link: [here](https://arxiv.org/pdf/2310.16810.pdf) | **AI Score:** :rocket: :rocket: |  **Interest Score:** :Magnet: | **Reading Time:** :alarm_clock:

                            **Result:** his paper explores the challenges of predicting individual uplifts and how multiple instance learning can help improve targeting for promotion campaigns. 
                            The authors identify the problem of counter-factual nature and fractional treatment effect in uplift modeling and 
                            propose a MIL-enhanced framework to accommodate two-model uplift methods. 
                            The framework uses a bag-wise loss to overcome the counter-factual problem and generates bags by 
                            clustering to overcome the fractional treatment effect problem. Experiments on two datasets 
                            suggest consistent improvements over existing SOTA methods.

                        """
                        },
                        {
                            "role": "user", # users prompt
                            "content": f'''
                                            here are the details of the paper {row_str}
                                        '''
                        }],
                        temperature=0.1,
                        max_tokens=4096,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )   
                    st.write(response.choices[0].message.content)

            

