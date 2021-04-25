from typing import List
import spacy
import string

from src.lm_core import GPTLM
from .utils import plot_topk, draw_wordcloud, scatter_plot

nlp = spacy.load("en_core_web_sm")
gpt = GPTLM()


def analyse_text(articles: List, filters=None, probs_plot=False):
    # from nltk.corpus import stopwords

    ################
    # import nltk
    # nltk.download('stopwords')
    ###############

    # curated stop_words
    if filters=="stopwords":
        with open("stopwords-en.txt") as f:
            content = f.readlines()
        stop_words = set([x.strip() for x in content] + list(string.punctuation))

    top_10_cnt, word_dict_10, probs_10 = 0, {}, []
    top_100_cnt, word_dict_100, probs_100 = 0, {}, []
    top_1000_cnt, word_dict_1000, probs_1000 = 0, {}, []
    top_x_cnt, word_dict_x, probs_x = 0, {}, []

    for article in articles:
        article_list = article.split("\n")

        # sen = ""
        # for s in article_list:
        #     if len((s + sen).split(' ')) < 800:
        #         sen = sen + " " + s
        # article = sen[1:]

        if filters=="ents":
            doc = nlp(article)
            ent_list = [ent.text for ent in doc.ents]

            np_list = [str(token) for token in doc if token.tag_[:2] == "NN"]

        for sentence in article_list:
        #for sentence in [article]:
            if len(sentence.split(" ")) > 5:
                text = gpt.tokenizer.bos_token + " " + sentence
                outputs = gpt.get_probabilities(text, top_k=1000)
                for idx, (rank, probs) in enumerate(outputs['true_topk']):
                    flag = True
                    if filters=="ents" and (outputs['bpe_strings'][idx + 1] not in ent_list): # or outputs['bpe_strings'][idx + 1] not in np_list):
                        flag = False
                    elif filters=="stopwords" and outputs['bpe_strings'][idx + 1] in stop_words:
                        flag = False

                    if flag:
                        if rank <= 10:
                            top_10_cnt += 1
                            word_dict_10[outputs['bpe_strings'][idx + 1]] = 1 - probs
                            # TODO: can include wordcloud for this too.
                            probs_10.append(outputs['pred_topk'][idx][0][1])
                        elif rank <= 100:
                            top_100_cnt += 1
                            word_dict_100[outputs['bpe_strings'][idx + 1]] = 1 - probs
                            probs_100.append(outputs['pred_topk'][idx][0][1])
                        elif rank <= 1000:
                            top_1000_cnt += 1
                            word_dict_1000[outputs['bpe_strings'][idx + 1]] = 1 - probs
                            probs_1000.append(outputs['pred_topk'][idx][0][1])
                        else:
                            top_x_cnt += 1
                            word_dict_x[outputs['bpe_strings'][idx + 1]] = 1 - probs
                            probs_x.append(outputs['pred_topk'][idx][0][1])

    data = {'top_10': top_10_cnt, 'top_100': top_100_cnt, 'top_1000': top_1000_cnt, 'top_x': top_x_cnt}
    plot_topk(data)
    draw_wordcloud(word_dict_10, word_dict_100, word_dict_1000, word_dict_x)

    if probs_plot:
        scatter_plot(probs_10)
        scatter_plot(probs_100)
        scatter_plot(probs_1000)
        scatter_plot(probs_x)
        #scatter_plot(probs_10, probs_100, probs_1000, probs_x)