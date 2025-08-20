from config import Config
from embeddings import Embedder

class VectorStore:
    def __init__(self, config: Config, embedder: Embedder):
        self.config = config
        self.embedder = embedder
        self.index = config.pc.Index(config.INDEX_NAME)

    def upsert_documents(self, docs):
        vectors = []
        for i, text in enumerate(docs, start=1):
            embedding = self.embedder.embed_text(text)
            vectors.append({
                "id": f"doc-{i}",
                "values": embedding,
                "metadata": {"text": text}
            })
        self.index.upsert(vectors=vectors)
        print(f"✅ {len(docs)} documents upserted into Pinecone")

# Example usage:
# if __name__ == "__main__":
#     docs = [
#         "The village of Brossenwald holds an annual competition where locals carve instruments out of ice and play them until they melt.",
#         "In the archives of the Marlowe Observatory, there is a log of an unexplained comet sighting from 1837 that appears in no modern astronomy catalog.",
#         "The mineral 'amberite' was once believed to glow faintly under moonlight, though modern tests suggest the effect was an optical illusion.",
#         "On the island of Verdanthold, farmers still use wind-powered seed scatterers that date back over 300 years.",
#         "The Clock of Drelmere runs backwards during the solstice, a phenomenon that has baffled horologists for centuries.",
#         "An old shipping ledger describes a cargo of 'whistling sand' that supposedly produced tones when poured from one vessel to another.",
#         "During the Festival of Lantern Shadows in Hallowmere, children wear masks painted with patterns that only reveal meaning under candlelight.",
#         "The lost manuscript of Ardin Telvash reportedly contains recipes for inks that never fade, even after centuries.",
#         "Travelers in the Marshes of Yorm speak of a moss that, when stepped on, produces a faint metallic ringing sound.",
#         "The town records of East Grindle mention a bridge built entirely of petrified wood, dismantled in 1892 after it was deemed too eerie."
#     ]
#     config = Config()
#     embedder = Embedder(config)
#     store = VectorStore(config, embedder)
#     store.upsert_documents(docs)