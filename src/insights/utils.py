import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import pandas as pd

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def plot_topk(data: dict, subtitile="", path="data/"):
    names = list(data.keys())
    values = list(data.values())

    fig, axs = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    bar = axs.bar(names, values)
    bar[0].set_color('green')
    bar[1].set_color('orange')
    bar[2].set_color('red')
    bar[3].set_color('black')
    fig.suptitle('Top k ranking for the original sequence in the predicted sequence filtered by removing stopwords')
    plt.show()
    plt.savefig(path + 'top_k.jpg')
    st.image(path + "top_k.jpg")


def draw_wordcloud(word_dict_10: dict, word_dict_100: dict, word_dict_1000: dict,
                   word_dict_x: dict, path='data/'):
    cloud10 = WordCloud(max_font_size=40, colormap="hsv").generate_from_frequencies(
        word_dict_10 if len(word_dict_10)!=0 else {"None": 1})
    cloud100 = WordCloud(max_font_size=40, colormap="hsv").generate_from_frequencies(
        word_dict_100 if len(word_dict_100)!=0 else {"None": 1})
    cloud1000 = WordCloud(max_font_size=40, colormap="hsv").generate_from_frequencies(
        word_dict_1000 if len(word_dict_1000)!=0 else {"None": 1})
    cloudx = WordCloud(max_font_size=40, colormap="hsv").generate_from_frequencies(
        word_dict_x if len(word_dict_x)!=0 else {"None": 1})

    fig, axs = plt.subplots(1, 4, figsize=(25, 20), sharey=True)

    axs[0].imshow(cloud10, interpolation='bilinear')
    axs[1].imshow(cloud100, interpolation='bilinear')
    axs[2].imshow(cloud1000, interpolation='bilinear')
    axs[3].imshow(cloudx, interpolation='bilinear')

    plt.axis('off')
    #plt.show()
    st.pyplot(fig)
    fig.set_figheight(50)
    fig.set_figwidth(50)

    for i in range(4):
        extent = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path + 'wordcloud_' + str(i) +'.jpg', bbox_inches=extent,
                    pad_inches=0)
    #fig.savefig(path + 'wordcloud.jpg')

#def scatter_plot(probs_10, probs_100, probs_1000, probs_x):
def scatter_plot(probs_10, col, path='data/scatter.jpg'):
    x, y = list(range(len(probs_10))), probs_10
    #x10, y10 = list(range(len(probs_10))), probs_10
    #x100, y100 = list(range(len(probs_100))), probs_100
    #x1000, y1000 = list(range(len(probs_1000))), probs_1000
    #xx, yx = list(range(len(probs_x))), probs_x


    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    #fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)
    #axs[0].scatter(x10, y10)
    #axs[1].scatter(x100, y100)
    #axs[2].scatter(x1000, y1000)
    #axs[3].scatter(xx, yx)

    ax_histx.hist(x)
    ax_histy.hist(y, orientation='horizontal')

    plt.show()
    with col:
        # st.header("")
        st.pyplot(fig)
    plt.savefig(path)
    #st.image("data/scatter.jpg")
