import pickle
from graph_tool.clustering import motifs, motif_significance

def main():
    with open("/home/usr8/15M54097/motifwalk/data/youtube_gt.data", "rb") as f:
        ytgt = pickle.load(f)
    yt_motif4_results = motif_significance(ytgt, k=4, n_shuffles=10, full_output=True)
    with open("./youtube_motif_results.pkl", "wb") as f:
        pickle.dump(yt_motif4_results, f)
    return 0

if __name__ == "__main__":
    main()
    
