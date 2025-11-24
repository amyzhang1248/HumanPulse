from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import tempfile 
from pytubefix import YouTube
import os
import random, string
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from utils import get_openai_api_key, init_pinecone_index, get_embedding, wordCount, scrapHFandCDC, splitContextIn3
from llmGenerator import generate_Gen_Qs, generate_summaries, generate_Relevant_Qs, generateSummariesList
import json 
import csv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=get_openai_api_key())
MODEL =  "gpt-3.5-turbo" # "gpt-4" 
TEMPERATURE = 0.7
perspectivePrefix = 'perspectives'
perLengthList = [100]
#read csv data file into dataframe
filenameIdentifier = 'split915'


#create and add questions to the dataframe
def addQuestions(df, start = 0):

    genericQList = []
    targetQList = []
    segmentedQList = []

    for i in range(len(df)):
        gQuestionList = []
        tQuestionList = []
        sQuestionList = []
        conversation = df['content'][i+ start]
        questions = generate_Gen_Qs(conversation, 5).split("\n")
        specQ1 = generate_Relevant_Qs(conversation, "statistics and number related questions", 3).split("\n")
        specQ2 = generate_Relevant_Qs(conversation, "method and/or results related questions", 3).split("\n")
        specQ3 = generate_Relevant_Qs(conversation, "insights and/or opinions related questions", 3).split("\n")
        gQuestionList.extend(questions)
        tQuestionList.extend(specQ1)
        tQuestionList.extend(specQ2)
        tQuestionList.extend(specQ3)
        segList = splitContextIn3(conversation)
        for index in range(len(segList)):
            questions = generate_Gen_Qs(segList[index], 2).split("\n")
            sQuestionList.extend(questions)
        
        filtered_list = [item for item in gQuestionList if item and item[0].isdigit()]
        genericQList.append(filtered_list)

        filtered_list = [item for item in tQuestionList if item and item[0].isdigit()]
        targetQList.append(filtered_list)

        filtered_list = [item for item in sQuestionList if item and item[0].isdigit()]
        segmentedQList.append(filtered_list)

    df['genericQuestions'] = genericQList
    df['targetQuestions'] = targetQList
    df['segmentQuestions'] = segmentedQList

def generateSummaries(df, startIndex =0):
    
    summaryList = []
    for i in range(len(df)):
        conversation = df['content'][i + startIndex]
        sumarries = generateSummariesList(conversation, 100)
        summaryList.append(sumarries)
    
    df['sumarries'] = summaryList



def createSavePerspectives(perLengthList, perFilename, records):
    for i in range(len(perLengthList)):
        print(i)
        for rIndex in range(len(records)):
            print(f"parLenth = {perLengthList[i]}, rindex = {rIndex}")
            conversation = records[rIndex]['content']
            perspectiveColumn = perspectivePrefix + str(perLengthList[i])
            summaries = generate_summaries(conversation, perLengthList[i])

            length = wordCount(summaries)
            records[rIndex][perspectiveColumn] = summaries

    with open(perFilename, 'w', encoding='utf-8') as outfile:
        json.dump(records, outfile, indent=4)

    


def update_pinecone_index(pinecone_index, perspectives, perspectiveColumn):

    vectors = []

    index = 0
    for perspective in perspectives:
        embedding = get_embedding(perspective[perspectiveColumn])
        vectors.append({
            "id": str(index),
            "values": embedding,
            "metadata": {
                "index": perspective['index'],
            }
        }
        )
        index += 1
    # Upsert vectors to Pinecone

    pinecone_index.upsert(vectors=vectors, namespace="perspectives")

def prepareData():
   #create documents 
   # filename ='database/all_candp_' + filenameIdentifier + '.json' 
   # readinData( filename, 20)
   # with open(filename, 'r') as f:
   #     records = json.load(f)

    perFilename = 'database/all_perspectives_' + filenameIdentifier + '.json'
   # createSavePerspectives(perLengthList, perFilename, records)

    with open(perFilename, 'r') as f:
        records = json.load(f)

    #Create a new pinecone index and save perspectives into pinecone index 
    for i in range(len(perLengthList)):
        perspectiveColumn = perspectivePrefix + str(perLengthList[i])
        pineconeIndexName = "perspective" + str(perLengthList[i])
        pinecone_index = init_pinecone_index(pineconeIndexName)
        update_pinecone_index(pinecone_index, records, perspectiveColumn )

#word count
def countPerspectives():

    perFilename = 'database/all_perspectives_' + filenameIdentifier + '.json'
    #createSavePerspectives(perLengthList, perFilename, records)
    with open(perFilename, 'r') as f:
        records = json.load(f)

    #Create a new pinecone index and save perspectives into pinecone index 
    count =[0 for _ in range(len(perLengthList))] 
    for i in range(len(perLengthList)):
        perspectiveColumn = perspectivePrefix + str(perLengthList[i])
        pineconeIndexName = "perspective" + str(perLengthList[i])
        for rIndex in range(len(records)):
             count[i] += wordCount(records[rIndex][perspectiveColumn])

    return count

def writeQandP():
       #read in datafile
    df = pd.read_csv('hf_bloglinks.csv')
    contentList = []
    recordsStart = 0
    recordsEnd = 486

    for index in range(recordsStart,recordsEnd ):
        url = df['link'][index]

        content = scrapHFandCDC(url)

        contentList.append(content)
    
    columns = ['title', 'link', 'content']

    # Create an empty DataFrame with specified column names
    new_df = pd.DataFrame(columns=columns)
    new_df['title'] = df['title'][recordsStart:recordsEnd]
    new_df['link'] = df['link'][recordsStart:recordsEnd]
    new_df['content'] = contentList
    #create 3 additional data columns
    addQuestions(new_df, recordsStart)

    #generate perspectives

    generateSummaries(new_df, recordsStart)


    json_data = new_df.to_json(orient="records")

    column_names = new_df.columns.tolist()
    
    jsonArray = []
      
    #convert each csv row into python dict
    for index in range(len(new_df)): 
        
        dictItem = {}
        #add this python dict to json array
        for itemIndex in range(len(column_names)):
            dictItem[column_names[itemIndex]] = new_df.iloc[index][itemIndex]
        jsonArray.append(dictItem)
  
    #convert python jsonArray to JSON String and write to file
    with open("hf_qandprompt_1.json", 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

    new_df.to_csv("hf_qandprompt_1.csv")

def uploadPineconeIndex(doc):
    
    pineconeIndexName = "hfperspectives"
    pinecone_index = init_pinecone_index(pineconeIndexName)
    
    vectors = []

    index = 0
  
    for index in range(len(doc)):


        perspectivesList = doc[index]['sumarries']

        for seq, value in enumerate(perspectivesList):
            print(index, seq)
            embedding = get_embedding(value)
            vectors.append({
                    "id": str(index) + str(seq),
                    "values": embedding,
                    "metadata": {
                        "index": index,
                        "sequence":seq
                    }
                }
            )
            
    # Upsert vectors to Pinecone
    pinecone_index.upsert(vectors=vectors, namespace="perspectives")


def uploadPineconeSingle(doc, segIndex, pineconeIndexName):
    
    pinecone_index = init_pinecone_index(pineconeIndexName)
    
    vectors = []

    index = 0
  
    for index in range(len(doc)):


        perspectivesList = doc[index]['sumarries']

        value = perspectivesList[segIndex]
        print(index, segIndex)
        embedding = get_embedding(value)
        vectors.append({
                "id": str(index) + str(segIndex),
                "values": embedding,
                "metadata": {
                    "index": index,
                    "sequence":segIndex
                }
            }
        )
            
    # Upsert vectors to Pinecone
    pinecone_index.upsert(vectors=vectors, namespace="perspectives")

def readinData():
    data = []
                     
    file_pathList = [    'qandprompt_1.json']
    
    for file in file_pathList:
        with open(file, 'r') as f:
            records = json.load(f)
            data.extend(records)

    return data

if __name__=='__main__':

 # create questions and perspectives
    writeQandP()
 
    doc = readinData()
 
    seglist = [4, 5]
    for seglistIndex in range(len(seglist)):
        pinecone_index_name = "hfperspectives_all"
        pinecone_index = init_pinecone_index(pinecone_index_name)
        uploadPineconeSingle(doc, seglist[seglistIndex], pinecone_index_name)
    
        questionsList = []

        columns = ['raw_index', 'raw_Q_seq', 'matched_index', 'matched_seq']
        rawIndexList = []
        rawQList = []
        matched_indexList = []
        matched_seqList = []

        totalMatches = 0
        totalQ = 0
        for index in range(len(doc)):
            print(index)
            questionsList.clear()
            questionsList.extend(doc[index]['genericQuestions'])
            questionsList.extend(doc[index]['targetQuestions'])
            questionsList.extend(doc[index]['segmentQuestions'])

            for seq, value in enumerate(questionsList):
                totalQ += 1
                response = pinecone_index.query(
                        namespace="perspectives",
                        vector=get_embedding(value),
                        top_k=1,
                        include_values=True,
                        include_metadata=True
                    )
                
                foundMatch:bool = False 
                for match in response['matches']:
                    metadata = match['metadata']
                    matchedIndex = metadata['index']
                    matchedSeq = metadata['sequence']
                    if matchedIndex == index:
                        matched_indexList.append(matchedIndex)
                        matched_seqList.append(matchedSeq)
                        foundMatch = True
                        totalMatches += 1
                        break
                if foundMatch == False:
                        matched_indexList.append(None)
                        matched_seqList.append(None)
                rawIndexList.append(index)
                rawQList.append(seq)

            
        columns = ['raw_index', 'raw_Q_seq', 'matched_index', 'matched_seq']
                
        new_df = pd.DataFrame(columns=columns)
        new_df['raw_index'] = rawIndexList
        new_df['raw_Q_seq'] = rawQList
        new_df['matched_index'] = matched_indexList
        new_df['matched_seq']  = matched_seqList

        print(totalQ)
        print(totalMatches)

        new_df.to_csv("results_allmatches_"+ str(seglist[seglistIndex]) + ".csv")
                    






    


    