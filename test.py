import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from gensim.summarization import summarize
import pandas as pd


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
    if choose == 'Text Summarization':
        activities = ["Summarize via Text", "Summazrize via File"]
        choice = st.selectbox("Select Activity", activities)
        if choice == "Summarize via Text":
            st.subheader("Summary using NLP")
            raw_text = st.text_area("Enter Text Here","Type here")
            summary_choice = st.selectbox("Summary Choice" , ["Gensim","Sumy Lex rank","NLTK"])
            if st.button("Summarize Via Text"):
                if summary_choice == 'Gensim':
                    summary_result = summarize(raw_text)

                elif summary_choice == 'Sumy Lex rank':
                    summary_result = summarize(raw_text)

                elif summary_choice == 'NLTK':
                    summary_result = summarize(raw_text)


                st.write(summary_result)

        if choice == "Summazrize via File":
            st.write("""
            ### Document Text Summary
            """)
            uploaded_file = st.file_uploader("Choose a Excel file")
            if uploaded_file is not None:
        #         bytes_data = uploaded_file.getvalue()
                data = pd.read_excel(uploaded_file)     
                st.write('Preview of the attached file',data.head(10))
                st.button("Submit")
    #         st.session_state["preview"] = data[:10]
    # #         for i in range(0, min(5, len(data))):
    # #             st.session_state["preview"] = st.session_state["preview"]+ data[i]
    #     preview = st.text_area("File Preview", "", height=150, key="preview")
    #     upload_state = st.text_area("Upload State", "", key="upload_state")
    #     def upload():
    #         if uploaded_file is None:
    #             st.session_state["upload_state"] = "Upload a file first!"
    #         else:
    #             data = uploaded_file.getvalue()
    #             parent_path = pathlib.Path(__file__).parent.parent.resolve()           
    #             save_path = os.path.join(parent_path, "data")
    #             complete_name = os.path.join(save_path, uploaded_file.name)
    #             destination_file = open(complete_name, "w")
    #             destination_file.write(data)
    #             destination_file.close()
    #             st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
    #     st.button("Upload file to Sandbox", on_click=upload)
    
if __name__ == '__main__':
    main()
