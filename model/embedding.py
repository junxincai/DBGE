from walker import HoraryWalker
from gensim.models import Word2Vec

class Embedding:
    def __init__(self, G, walk_length, num_walks, workers=1):
        self.G = G
        self.model = None
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_nodes, self.item_nodes = self.read_nodes()
        self.walker = HoraryWalker(G, self.user_nodes)
        self.sentences = self.walker.assign_walks(num_walks=num_walks, walk_length=walk_length, workers=workers,
                                                    verbose=1)

    def read_nodes(self):
        user_nodes = []
        f = open('../data/Amazon/amazon_user.txt', 'r')
        for line in f.readlines():
            user_nodes.append(line.strip())
        f.close()

        item_nodes = []
        f = open('../data/Amazon/amazon_item.txt', 'r')
        for line in f.readlines():
            item_nodes.append(line.strip())
        f.close()
        return user_nodes, item_nodes

    def train(self, embed_size=128, window_size=5, iter=5, workers=1, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 1
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Embedding...")
        self.model  = Word2Vec(**kwargs)
        print("Embedding... done")

        return self.model

    def get_embeddings(self):
        if self.model is None:
            print("model not train")
            return {}

        for user_node in self.user_nodes:
            self.user_embeddings[user_node] = self.model.wv[user_node].tolist()
        for item_node in self.item_nodes:
            self.item_embeddings[item_node] = self.model.wv[item_node].tolist()
        return self.user_embeddings, self.item_embeddings
