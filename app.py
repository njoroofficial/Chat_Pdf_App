import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import CharacterTextSplitter # help divide text into chunks
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# a database to store our vectors (faiss)
from langchain.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory



class chat_PDF:

    #function to read and extract text from documen
    def get_pdf_text(self,pdf_docs):
        # Initialize an empty string to store the text content
        text = ""
        # loop through the pdfs
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            # loop through the pages and extract text
            for pages in pdf_reader.pages:
                # Extract the text from the page
                text += pages.extract_text()

        return text
    
    # function for book review
    def book_Preview(self, pdf_docs):
            st.header('Document Preview :books:')
            raw_text = chatPDF.get_pdf_text(pdf_docs)
            st.write(raw_text)
            


    # function to split text into chunks and return as list
    def get_text_chunks(self,text):
        # create instance
        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
        )
        chunks = text_splitter.split_text(text)
        return chunks


    def get_vectorstore(self,text_chunks):
        embeddings = OpenAIEmbeddings()
        #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        # we generate our databse from text
        vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings) 
        return vectorstore


    def get_conversation_chain(self,vectorstore):
        llm = ChatOpenAI()

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = vectorstore.as_retriever(),
            memory = memory,
            )
        return conversation_chain

    # to get rensponse to user question
    def handle_userinput(self,user_question):
        if st.session_state.conversation:

            response = st.session_state.conversation({'question':user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.warning('Conversation not initialized. Upload and process document first.') 



    def main(self):
        
        load_dotenv() # help langchain access our API KEYS

        st.set_page_config(page_title='Chat With Pdfs',page_icon=':books:')

        # adding css
        st.write(css, unsafe_allow_html=True)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        # our header_title
        st.header(':red[Large Language Model] :spider::robot_face:')

        # for our textarea
        user_question = st.text_input('Ask a question about your documents:')

        if user_question:
            self.handle_userinput(user_question)       

    
        #sidebar for uploading documents
        with st.sidebar:
            st.subheader('Your Documents :books:')
            pdf_docs = st.file_uploader("Upload your document and click 'Process'", accept_multiple_files=True)
        
            if st.button('Process'):
                with st.spinner('In Progress....'):
                    # get the document text
                    raw_text = self.get_pdf_text(pdf_docs)
                    #st.write(raw_text)

                    # get the text chunk
                    text_chunks = self.get_text_chunks(raw_text) # return a list of text in chunks
                    #st.write(text_chunks)

                    # create vector store/ embeddings
                    vectorstore = self.get_vectorstore(text_chunks)

                    # create conversation_chain
                    st.session_state.conversation = self.get_conversation_chain(vectorstore)
            if st.button('Preview'):
                with st.spinner('Book Preview Loading...'):
                    self.book_Preview(pdf_docs)

    
# creating objects of this classes
chatPDF = chat_PDF()

chatPDF.main()