import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from gensim.summarization import summarize
import pandas as pd
import numpy as np
from io import BytesIO
import time
from pyxlsb import open_workbook as open_xlsb
from transformers import *
from sentence_transformers import SentenceTransformer, util
from summarizer import Summarizer
import torch
import nltk
from scipy.sparse.csgraph import connected_components
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.cache(allow_output_mutation=True)
def load_model():
    nltk.download('punkt')
    custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    return model

@st.cache(allow_output_mutation=True)
def lex_customized():  
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def main():
    lex_model = lex_customized()
    bert_model = load_model()
    
    def lex_model_main(text):
        def degree_centrality_scores(
            similarity_matrix,
            threshold=None,
            increase_power=True,):
            if not (threshold is None
                or isinstance(threshold, float)
                and 0 <= threshold < 1):

                raise ValueError('\'threshold\' should be a floating-point number ''from the interval [0, 1) or None',)

            if threshold is None:
                markov_matrix = create_markov_matrix(similarity_matrix)
            else:
                markov_matrix = create_markov_matrix_discrete(
                    similarity_matrix,
                    threshold,)

            scores = stationary_distribution(
                markov_matrix,
                increase_power=increase_power,
                normalized=False,)
            return scores

        def _power_method(transition_matrix, increase_power=True):
            eigenvector = np.ones(len(transition_matrix))
            if len(eigenvector) == 1:
                return eigenvector
            transition = transition_matrix.transpose()
            while True:
                eigenvector_next = np.dot(transition, eigenvector)
                if np.allclose(eigenvector_next, eigenvector):
                    return eigenvector_next
                eigenvector = eigenvector_next
                if increase_power:
                    transition = np.dot(transition, transition)

        def connected_nodes(matrix):
            _, labels = connected_components(matrix)
            groups = []
            for tag in np.unique(labels):
                group = np.where(labels == tag)[0]
                groups.append(group)
            return groups

        def create_markov_matrix(weights_matrix):
            n_1, n_2 = weights_matrix.shape
            if n_1 != n_2:
                raise ValueError('\'weights_matrix\' should be square')
            row_sum = weights_matrix.sum(axis=1, keepdims=True)
            return weights_matrix / row_sum

        def create_markov_matrix_discrete(weights_matrix, threshold):
            discrete_weights_matrix = np.zeros(weights_matrix.shape)
            ixs = np.where(weights_matrix >= threshold)
            discrete_weights_matrix[ixs] = 1
            return create_markov_matrix(discrete_weights_matrix)

        def graph_nodes_clusters(transition_matrix, increase_power=True):
            clusters = connected_nodes(transition_matrix)
            clusters.sort(key=len, reverse=True)
            centroid_scores = []
            for group in clusters:
                t_matrix = transition_matrix[np.ix_(group, group)]
                eigenvector = _power_method(t_matrix, increase_power=increase_power)
                centroid_scores.append(eigenvector / len(group))
            return clusters, centroid_scores

        def stationary_distribution(
            transition_matrix,
            increase_power=True,
            normalized=True,):
            n_1, n_2 = transition_matrix.shape
            if n_1 != n_2:
                raise ValueError('\'transition_matrix\' should be square')
            distribution = np.zeros(n_1)
            grouped_indices = connected_nodes(transition_matrix)
            for group in grouped_indices:
                t_matrix = transition_matrix[np.ix_(group, group)]
                eigenvector = _power_method(t_matrix, increase_power=increase_power)
                distribution[group] = eigenvector
            if normalized:
                distribution /= n_1
            return distribution
        
        sentences = nltk.sent_tokenize(text)
        embeddings = lex_model.encode(sentences, convert_to_tensor=True)         #Compute the sentence embeddings
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()         #Compute the pair-wise cosine similarities
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)        #Compute the centrality for each sentence
        most_central_sentence_indices = np.argsort(-centrality_scores)        #We argsort so that the first element is the sentence with the highest score
        summary_list=[]                                                     #return 10 sentences with the highest scores
        for idx in most_central_sentence_indices[0:10]:
            summary_list.append(sentences[idx])
        return (' '.join(summary_list)) 
    
    def bert_model_main(text):
        summary = bert_model(text)
        if summary!='':
            return summary
        else:
            return text
    
    st.title("Text Summarizer App")    
    with st.sidebar:
        choose = option_menu("App Gallery", ["About", "Text Summarization", "Topic Modelling",  "Contact"],
                             icons=['house-fill', 'book-half', 'badge-tm-fill','person-lines-fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
        )

    if choose == 'Text Summarization':
        activities = ["Summarize via Text", "Summazrize via File"]
        choice = st.selectbox("Select Activity", activities)
        if choice == "Summarize via Text":
            st.subheader("Summary using NLP")
            raw_text = st.text_area("Enter Text Here","Type here")
            summary_choice = st.selectbox("Summary Choice" , ["Gensim","Lex Rank","BERT"])
            if st.button("Summarize Via Text"):
                if summary_choice == 'Gensim':
                    try:
                        summary_result = summarize(raw_text)
                    except:
                        summary_result = raw_text
                elif summary_choice == 'Lex Rank':
                    summary_result = summarize(raw_text)

                elif summary_choice == 'BERT':
                    with st.spinner('Processing...'):
                        summary_result = bert_model_main(raw_text)

                st.write(summary_result)

        if choice == "Summazrize via File":
            st.write("""
            ### Document Text Summary
            """)

            uploaded_file = st.file_uploader("Choose a Excel file")
            if uploaded_file is not None:
                with st.spinner('Wait for it...'):    
                    data = pd.read_excel(uploaded_file)     
                    st.write('Preview of the attached file',data.head(10))
                    col1, col2 = st.columns([1,1])
                    with col1:
                        column_choice = st.selectbox("Select Column" , data.columns.to_list())
                    with col2:
                        summary_choice = st.selectbox("Summary Choice" , ["Gensim","Lex Rank","BERT"])
                col3, col4 = st.columns([1,6])
                with col3:
                    submit_data = st.button("Submit")
                    if submit_data:                                       
                        with col4:
                            with st.spinner('Processing...'):
                                def get_summary(data,column_choice,summary_choice):
                                    if summary_choice == 'Gensim':
                                        try:
                                            summary_result = summarize(raw_text)
                                        except:
                                            summary_result = raw_text
                                    elif summary_choice == 'Lex Rank':
                                        data = data.replace(np.nan,'')
                                        data[column_choice+' Summary'] = data[column_choice].apply(lex_model_main)
                                    elif summary_choice == 'BERT':
                                        data = data.replace(np.nan,'')
                                        data[column_choice+' Summary'] = data[column_choice].apply(bert_model_main)
                                    return data                              

                                def to_excel(df):
                                    output = BytesIO()
                                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                                    workbook = writer.book
                                    worksheet = writer.sheets['Sheet1']
                                    format1 = workbook.add_format({'num_format': '0.00'}) 
                                    worksheet.set_column('A:A', None, format1)  
                                    writer.save()
                                    processed_data = output.getvalue()
                                    return processed_data

                                data = get_summary(data,column_choice,summary_choice)
                                df_xlsx = to_excel(data)
                                st.download_button(label='📥 Download Processed Result',
                                                                data=df_xlsx ,
                                                                file_name= 'df_test.xlsx')
#                                 st.button("Download Processed file")
#                             st.text("Processed Successfully")
    
if __name__ == '__main__':
    main()
