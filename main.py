import streamlit as st
from src.insights import topk_insight
import pandas as pd

from pathlib import Path


def main():
    st.sidebar.header("Configuration")

    txt = st.sidebar.text_area('Text to Analyze')
    dataset_list = ['NewsCorpus-2020', 'Arxiv', 'ADM_articles','ADM_tickets']
    dataset = st.sidebar.selectbox("or pick a dataset", ['None'] +
                                   dataset_list)
    filter = st.sidebar.selectbox("Add filters, If necessary!", ('None', 'Stopwords',
                                                                 'ents'))
    starting_idx = st.sidebar.slider('customize starting index for analysis in each '
                                     'content', 0, 512, 0)
    if dataset == "ADM_articles":
        df = pd.read_pickle("data/ADM_articles/ADM_articles_dec.pkl")['content'].to_list()
        document_count = st.sidebar.slider('How many documents do you want to '
                                           'analyse?', 0, len(df), 10)
        df = df[:document_count]
        path = "data/ADM_articles/" + str(document_count) + "_" + str(filter) + "_" + \
               str(starting_idx) + "_"
    elif dataset == "ADM_tickets":
        df = pd.read_pickle("data/ADM_tickets/ADM_tickets_dec.pkl")
        df = (df['summary'] + ". " + df['description']).to_list()
        issues_count = st.sidebar.slider('How many issues do you want to '
                                         'analyse?', 0, len(df), 10)
        df = df[:issues_count]
        path = "data/ADM_tickets/" + str(issues_count) + "_" + filter + "_" + \
               str(starting_idx) + "_"

    else:
        path = "data/"

    st.header("You can also pick an precomputed experiment to see some good charts!")
    experiment_list = ["100 ADM articles - ents + NP filtered"]
    experiment = st.selectbox("Select Experiment:", options=[None] + experiment_list,
                              index=0)
    if dataset in dataset_list:
        st.sidebar.text("Show pre-computed results, if exists?")
        yes = st.sidebar.checkbox("yes")
        no = st.sidebar.checkbox("no")
    if st.sidebar.button('Analyse Text') or st.button('Show Experiment results!'):
        if dataset in dataset_list or experiment!=None:
            if experiment!=None:
                if experiment_list.index(experiment) == 0:
                    path = "data/ADM_tickets/" + str(10) + "_ents_0_"
                    yes, no = True, False
            if Path(path + "top_k.jpg").is_file():
                if yes:
                    with st.spinner("Fetching results..."):
                        st.image(path + "top_k.jpg")
                        col1, col2, col3, col4 = st.beta_columns(4)
                        with col1:
                            st.image(path + "wordcloud_0.jpg")
                        with col2:
                            st.image(path + "wordcloud_1.jpg")
                        with col3:
                            st.image(path + "wordcloud_2.jpg")
                        with col4:
                            st.image(path + "wordcloud_3.jpg")

                        col1, col2, col3, col4 = st.beta_columns(4)
                        with col1:
                            st.image(path + "scatter_col1.jpg")
                        with col2:
                            st.image(path + "scatter_col2.jpg")
                        with col3:
                            st.image(path + "scatter_col3.jpg")
                        with col4:
                            st.image(path + "scatter_col4.jpg")
                if no:
                    with st.spinner("Computing Results..."):
                        topk_insight.analyse_text(df, filters=filter, probs_plot=True,
                                                  starting_idx=starting_idx,
                                                  path=path)
            else:
                with st.spinner("Computing Results..."):
                    topk_insight.analyse_text(df, filters=filter, probs_plot=True,
                                              starting_idx=starting_idx,
                                              path=path)

        else:
            topk_insight.analyse_text([txt], filters=filter, probs_plot=True,
                                      starting_idx=starting_idx,
                                      path=path)

main()