try:
    from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
    print("Found in langchain_community.graphs.graph_document")
except ImportError:
    try:
        from langchain_core.graph_vectorstores.base import GraphDocument, Node, Relationship
        print("Found in langchain_core.graph_vectorstores.base")
    except ImportError:
        print("Could not find GraphDocument")
