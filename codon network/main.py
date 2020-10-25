from utils import *
from gensim.models import Word2Vec

if __name__ == '__main__':
    G = createGraph(codon2AA, rate)
    walks = build_Deepwalk_corpus(G, num_paths=100, path_length=4)
    model = Word2Vec(walks, size=50, window=3, min_count=0, sg=1, hs=1, workers=4)
    model.wv.save_word2vec_format('output.txt')
    data = model.wv.vectors
    labels = model.wv.index2word
    distance = compute_Codon_distance(labels,data)
    pro2codon = generate_Pro2codon(codon2AA)
    pro_dist = compute_Pro_distance(distance, pro2codon)
    for (pro1, pro2), dist in pro_dist.items():
        print('Protein Distance %s-%s %.3f' % (pro1, pro2, dist))
    dist_matrix = dict2matrix(pro_dist)
    labels = sorted(list(pro2codon.keys()))
    plot(dist_matrix, labels)
    