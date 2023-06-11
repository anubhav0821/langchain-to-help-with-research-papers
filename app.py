import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey


#App framework
st.title('🦜🔗 LangChain to help with research papers.')
prompt = st.text_input('Plug in your prompt here')

#create prompt template
abstract_template = PromptTemplate(
    input_variables = ['topic'],
    template='write be a abstraction research paper abstract on {topic} and try not to hallucinate if you do not know the answer to the question asked by the user. Just ask for more context if that is the case.'
)

explanation_template = PromptTemplate(
    input_variables = ['abstract', 'wikipedia_research'],
    template='write an detailed explanation with no less that a 1500 words that can be used to understand the topic and write research paper on the abstract abstract : {abstract}, while leveraging and understanding this wikipedia research:{wikipedia_research}'
)


# Memory to store memory
abstract_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
explanation_memory = ConversationBufferMemory(input_key='abstract', memory_key='chat_history')

# llms
llm = OpenAI(temperature=0.9)
abstract_chain = LLMChain(llm=llm, prompt=abstract_template, verbose=True, output_key='abstract', memory=abstract_memory)
explanation_chain = LLMChain(llm=llm, prompt=explanation_template, verbose=True, output_key='explanation', memory=explanation_memory)

wiki = WikipediaAPIWrapper()

#display the prompt if there is a responce
if prompt:
    abstract = abstract_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    explanation = explanation_chain.run(abstract=abstract, wikipedia_research = wiki_research )
    st.write(abstract)
    st.write(explanation)

    with st.expander('Abstract'):
        st.info(abstract_memory.buffer)

    with st.expander('Detailed explananion'):
        st.info(explanation_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

    

