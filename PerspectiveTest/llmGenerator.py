import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
#from utils import get_openai_api_key
import json
import os
import prompts

load_dotenv()

#os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
#client = OpenAI(api_key=get_openai_api_key())
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

MODEL =  "gpt-4" #"gpt-3.5-turbo"  #"gpt-4" "gpt-3.5-turbo"
TEMPERATURE = 0.7


def generate_summaries(content, wordCount = 100):

    combined_text = content #"\n\n".join(content)

    summarize_prompt = f""" Convert the following content into a technical summary of using {wordCount} completion_tokens or less that highlights the key achievements,   
      lessons learned, and actionable insights that can benefit others in the field.  Focus on the technical 
      aspects of the work, the impact of the work on the field, and how others can apply these insights. 
      Ensure that the summary is professional and geared towards an audience looking to gain knowledge or 
       apply similar approaches in their own work or looking for future research topics. 
       Make the summary a 1-5 sentence self-explanatory paragraph.  \n\n{combined_text}"""

    messages = [
        {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
        {"role": "user", "content": summarize_prompt}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)

    summary =  response.choices[0].message.content

    return summary


def generate_answers(content, question, wordCount = 100):

    combined_text = content #"\n\n".join(content)

    summarize_prompt = f""" answer this {question} according to the content given content: using {wordCount}. 
      please go direct to answer the questions. Do not include any thing like this article, this blog, etc. Straight forward to the answer.
      please use original text and writing style as much as possible and focus on  key achievements,   
      lessons learned, and actionable insights that can benefit others in the field.  Focus on the technical 
      aspects of the work, the impact of the work on the field, and how others can apply these insights. 
      Ensure that the answer is professional and geared towards an audience looking to gain knowledge or 
      apply similar approaches in their own work or looking for future research topics. Please remember to maintain personality
       \n\n{combined_text}"""

    messages = [
        {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
        {"role": "user", "content": summarize_prompt}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE,  max_tokens=wordCount)

    summary =  response.choices[0].message.content

    return summary

def generateSummariesList(content, wordCount = 100):

    combined_text = content #"\n\n".join(content)

    promptList = prompts.createPromptList()
    #promptList = prompts.create2PromptList()

    summaryList = []
    for index in range(len(promptList)):

        summarize_prompt = promptList[index].replace("wordCount", "strict" + str(wordCount)) + f"""\n\n{combined_text}"""
        messages = [
            {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
            {"role": "user", "content": summarize_prompt}
        ]

        response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)

        summary =  response.choices[0].message.content
        summaryList.append(summary)

    return summaryList


def shorten_summaries(content, wordCount = 100):

    shorten_prompt = f""" shorten the following summary into a maxinum {wordCount} words paragraph and keep the key achievements,   
      lessons learned, and actionable insights that can benefit others in the field.  Focus on the technical 
      aspects of the work, the impact of the work on the field, and how others can apply these insights. 
      Ensure that the summary is professional and geared towards an audience looking to gain knowledge or 
       apply similar approaches in their own work or looking for future research topics. \n\n{content}"""

    messages = [
        {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
        {"role": "user", "content": shorten_prompt}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)

    return response.choices[0].message.content

def generate_Gen_Qs(content, count = 10):

    combined_text = content #"\n\n".join(content)

    summarize_prompt = f""" create and list {count} technical questions from the content below: \n\n{content}"""

    messages = [
        {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
        {"role": "user", "content": summarize_prompt}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)

    return response.choices[0].message.content

def generate_Relevant_Qs(content, spec, count = 1):

    combined_text = content #"\n\n".join(content)

    summarize_prompt = f""" create and list {count} {spec} from the content below: \n\n{content}"""

    messages = [
        {"role": "system", "content": "You are a helpful and trustworthy assistant. You will generate an answer using the provided contexts."},
        {"role": "user", "content": summarize_prompt}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)

    return response.choices[0].message.content

def processJsonData(filepath, numberofrecords):

    with open(filepath, 'r') as f:
        data = json.load(f)

    toprow = min(len(data), numberofrecords)

    outdata = []
    for i in range(toprow):
        print(i)
        perspective = generate_summaries(data[i]['content'])
        outdata.append(data[i])
        outdata[i]['perspectives'] = perspective

    return outdata
 



    
if __name__=='__main__':
   
    # retrieving columns by indexing operator
   #medium 
   # dataFile = 'database/medium_data.json'
   # outputFile = 'database/medium_data_per_20_100W.json'

   #quora
   # dataFile = 'database/quora_data.json'
   # outputFile = 'database/quora_data_per_20_100W.json'

   # youtube
    dataFile = 'database/youtube_data.json'
    outputFile = 'database/youtube_data_per_20_100W.json'

    maxrecords = 20
    data = processJsonData(dataFile, maxrecords)

    with open(outputFile,'w') as file:
        json.dump(data, file, indent = 4)