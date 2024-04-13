import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function to prompt and get response from LLAma 2 model

def getLLamaresponse(input_text, no_words, blog_style):

    ### LLama2 model
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens': 256,
                              'temperature': 0.01})

    ## Template for the prompt

    template = """
        Write an explanation for {blog_style} on the topic of {input_text}
        within {no_words} words.
            """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                          template=template)

    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response


st.set_page_config(page_title="Ask LLaMA",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("ASK LLaMA 🤖")

input_text = st.text_input("Enter the topic you want me to explain")

## creating two more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the explanation for',
                            ('Common People', 'Researchers', 'Data Scientist'), index=0)

submit = st.button("Generate")

## response from LLaMA
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)
