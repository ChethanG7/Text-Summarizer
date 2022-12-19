import streamlit as st
from gensim.summarization import summarize

def main():
    
    st.title("Text Summarizer App")
    
    activities = ["Summarize Via Text"]#, "Summazrize via URL"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    if choice == 'Summarize Via Text':
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


    st.write("""
    # File Picker
    """)
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = uploaded_file.getvalue().splitlines()         
        st.session_state["preview"] = ''
        for i in range(0, min(5, len(data))):
            st.session_state["preview"] += data[i]
    preview = st.text_area("CSV Preview", "", height=150, key="preview")
    upload_state = st.text_area("Upload State", "", key="upload_state")
    def upload():
        if uploaded_file is None:
            st.session_state["upload_state"] = "Upload a file first!"
        else:
            data = uploaded_file.getvalue()
            parent_path = pathlib.Path(__file__).parent.parent.resolve()           
            save_path = os.path.join(parent_path, "data")
            complete_name = os.path.join(save_path, uploaded_file.name)
            destination_file = open(complete_name, "w")
            destination_file.write(data)
            destination_file.close()
            st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
    st.button("Upload file to Sandbox", on_click=upload)
    
if __name__ == '__main__':
    main()
