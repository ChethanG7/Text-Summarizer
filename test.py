import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from gensim.summarization import summarize
import pandas as pd
from io import BytesIO
import time
from pyxlsb import open_workbook as open_xlsb
from transformers import *
from summarizer import Summarizer


def main():
    
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
        
    def bert_custom_model(text):
        
        # Load model, model config and tokenizer via Transformers
        custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
        custom_config.output_hidden_states=True
        custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)
        model = Summarizer()
        return model(body)
    
    if choose == 'Text Summarization':
        activities = ["Summarize via Text", "Summazrize via File"]
        choice = st.selectbox("Select Activity", activities)
        if choice == "Summarize via Text":
            st.subheader("Summary using NLP")
            raw_text = st.text_area("Enter Text Here","Type here")
            summary_choice = st.selectbox("Summary Choice" , ["Gensim","Sumy Lex rank","BERT"])
            if st.button("Summarize Via Text"):
                if summary_choice == 'Gensim':
                    try:
                        summary_result = summarize(raw_text)
                    except:
                        summary_result = raw_text
                elif summary_choice == 'Sumy Lex rank':
                    summary_result = summarize(raw_text)

                elif summary_choice == 'BERT':
                    summary_result = bert_custom_model(raw_text)
                    
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
                        summary_choice = st.selectbox("Summary Choice" , ["Gensim","Sumy Lex rank","BERT"])
                col3, col4 = st.columns([1,6])
                with col3:
                    submit_data = st.button("Submit")
                    if submit_data:                                       
                        with col4:
                            with st.spinner('Processing...'):
                                def get_summary(data,column_choice,summary_choice):
                                    
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
