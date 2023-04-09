import numpy as np
import matplotlib.pyplot as plt
import os
import random

max_len = 12
category_names = [str(i) for i in range(max_len)]

results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5,0,0]
}

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


if __name__ == '__main__':
    dst_dir = './vis_fea'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    badcase_list = np.load('badcase.npy')
    badcase_list = ['./data/test/ir_front/fatigue_fanzhikang_1599750178_ir_front__7868.txt']


    # dst_dir = './vis_fea_normal'
    # src_dir = './data/test/ir_front'
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # badcase_list = os.listdir(src_dir)
    # badcase_list = [os.path.join(src_dir,v) for v in badcase_list if 'fanzhikang' in v]

    np.random.shuffle(badcase_list)
    for badcase_txt in badcase_list:
        with open(badcase_txt, 'r') as f:
            fea_info = f.read().strip().split()
            fea_info = list(map(float,fea_info))
        label = fea_info[-2]

        if label == 0 and random.random() < 0.9:
            continue

        debug = 0

        feas = {'bink_bin':fea_info[:11],
                'average_blink_duration': [fea_info[11]/20],
                'average_closed_duration': [fea_info[12]/10],
                'micro_sleep': [fea_info[13]/5],
                'average_close_duration': [fea_info[14]/10],
                'average_open_duration': [fea_info[15]/10],
                }

        for k,v in feas.items():
            if len(v) < max_len:
                v.extend([0 for _ in range(max_len - len(v))])

            v = [vv *100 for vv in v]
            feas[k] = v


        survey(feas, category_names)

        plt_name = str(int(label)) + '_' + os.path.basename(badcase_txt).replace('.txt','.png')
        plt.savefig(os.path.join(dst_dir,plt_name), dpi=300)
        plt.cla()





