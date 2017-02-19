import mpltex
import numpy as np
import matplotlib.pyplot as plt

@mpltex.acs_decorator
def plot_sigm(fname='motif_zscore_d.csv'):
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    labels = ['Cora', 'Citeseer', 'PubMed', 'NELL', 'Blogcatalog', 'PPI']
    score = data[:,1:].T  # Ignore motif id column and transpose
    print(score.shape)
    x = range(1,12)  # 11 motifs
    fig, ax = plt.subplots(1, figsize=(9,3))

    # The default line style is iterating over
    # color, line, and marker with hollow types.
    linestyles = mpltex.linestyle_generator()
    for i in range(score.shape[0]):
        ax.plot(x, score[i], **linestyles.__next__(),
                linewidth=2.4, markersize=7)
    ax.yaxis.grid(True)
    ax.set_xticklabels([])
    fig.tight_layout(pad=0.1)
    fig.savefig('./sigm_nolegend.pdf')

def main():
    plot_sigm()

if __name__ == '__main__':
    main()
