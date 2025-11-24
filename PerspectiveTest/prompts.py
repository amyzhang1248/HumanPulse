
def create2PromptList():

    promptList = []

    
    promptList.append(f"""Generate a technical summary of the content in wordCount completion_tokens or fewer.
                       Emphasize essential statistics that offer context, such as sample sizes, key metrics, 
                      or error rates, as well as any significant equations or mathematical models used in the analysis.""")
    
    promptList.append(f"""Produce a technical summary of the following content in wordCount completion_tokens 
                      or fewer, focusing on actionable insights and implementation strategies. 
                      Where applicable, include practical applications of the findings and discuss 
                      how they can be utilized in real-world scenarios""")
    
    
    return promptList



def createPromptList():

    promptList = []

    promptList.append(f"""Summarize the following content in  wordcount completion_tokens or fewer, emphasizing the main technical 
                      achievements, lessons learned, and actionable insights that are beneficial to others in the field. Focus on
                       how the work impacts the industry, its technical contributions, and practical applications. The summary 
                      should be professional, concise (1-5 sentences), and geared toward an audience interested in knowledge transfer, 
                      similar methodologies, or potential research directions.""")
    
    promptList.append(f"""Act as an academic expert, synthesize the following content into a concise, 
                      clear summary that encapsulates the main findings, methodologies, results, 
                      and implications. Retain essential terms and nuances. The summary sould be 
                      in wordcount completion_tokens or fewer""")
        
    promptList.append(f"""Create a professional summary of the following content in wordcount completion_tokens or less, 
                        capturing the essential insights and guidance for technical audience. Include core technical terms
                        (e.g., “backpropagation,” “non-linear regression”) and clarify any uncommon terms specific to this field, 
                        ensuring the author’s message is accurately conveyed.""")
    

    promptList.append(f"""Acting as an academic expert, propose a focused research topic derived from the content 
                      provided. The topic should address a gap or opportunity in current knowledge, with relevance 
                      to ongoing discussions or societal needs. Consider key variables, methods, and outcomes.
                      The summary sould be in wordcount completion_tokens or fewer""")
    
    promptList.append(f"""Generate a technical summary of the content in wordCount completion_tokens or fewer.
                       Emphasize essential statistics that offer context, such as sample sizes, key metrics, 
                      or error rates, as well as any significant equations or mathematical models used in the analysis.""")
    
    promptList.append(f"""Produce a technical summary of the following content in wordCount completion_tokens 
                      or fewer, focusing on actionable insights and implementation strategies. 
                      Where applicable, include practical applications of the findings and discuss 
                      how they can be utilized in real-world scenarios""")
    
    promptList.append(f"""Locate and list one or two tangential or unrelated viewpoints within the article, 
                      noting their placement (e.g., beginning, middle or end). Identify sections that do not directly support the main arguments, 
                      such as off-topic comments, unusual anecdotes, or loosely related statements. It sould be in wordcount completion_tokens or fewer """)

    return promptList

