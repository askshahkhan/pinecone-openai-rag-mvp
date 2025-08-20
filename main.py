from config import Config
from embeddings import Embedder
from vectorstore import VectorStore
from query import QueryEngine

if __name__ == "__main__":
    # Initialize components
    config = Config()
    embedder = Embedder()
    store = VectorStore(config, embedder)

    # Example documents
    docs = [
        "The village of Brossenwald holds an annual competition where locals carve instruments out of ice and play them until they melt.",
        "In the archives of the Marlowe Observatory, there is a log of an unexplained comet sighting from 1837 that appears in no modern astronomy catalog.",
        "The mineral 'amberite' was once believed to glow faintly under moonlight, though modern tests suggest the effect was an optical illusion.",
        "On the island of Verdanthold, farmers still use wind-powered seed scatterers that date back over 300 years.",
        "The Clock of Drelmere runs backwards during the solstice, a phenomenon that has baffled horologists for centuries.",
        "An old shipping ledger describes a cargo of 'whistling sand' that supposedly produced tones when poured from one vessel to another.",
        "During the Festival of Lantern Shadows in Hallowmere, children wear masks painted with patterns that only reveal meaning under candlelight.",
        "The lost manuscript of Ardin Telvash reportedly contains recipes for inks that never fade, even after centuries.",
        "Travelers in the Marshes of Yorm speak of a moss that, when stepped on, produces a faint metallic ringing sound.",
        "The town records of East Grindle mention a bridge built entirely of petrified wood, dismantled in 1892 after it was deemed too eerie."
    ]

    # Upsert docs
    store.add_documents(docs)

    # Ask a question
    engine = QueryEngine(store)
    question = "What is special about the Clock of Drelmere?"
    answer = engine.ask(question)
    print("Q:", question)
    print("A:", answer)