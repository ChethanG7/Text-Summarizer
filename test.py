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


if __name__ == '__main__':
    main()
