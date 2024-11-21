import os

import docker
import requests
# import streamlit as st
# from chains import load_embedding_model
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph

from chains import load_embedding_model
# from PIL import Image
# from streamlit.logger import get_logger
from utils import create_constraints, create_vector_index

load_dotenv(".env")

url = "bolt://localhost:7687"
username = "neo4j"
password = "password"
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
# create_constraints(neo4j_graph)
# create_vector_index(neo4j_graph, dimension)
# embeddings, dimension = load_embedding_model(
#     embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
# )
client = docker.from_env()
# load docker data on the basis of
def load_docker_data():
    

    for container in client.containers.list():
        # nameemb = embeddings.embed_query(container.name)
        # iamge = embeddings.embed_query(container.image)
        # nameemb = embeddings.embed_query(container.name)
        insert_docker_data({"name": container.name, "status":container.status, "image": container.image.id, "stats": container.stats(stream=False)["cpu_stats"]["system_cpu_usage"]/10**12})
    


def insert_docker_data(data: dict) -> None:
    
    
    import_query = """
    UNWIND $data AS item
    MERGE (containerInstance:ContainerInstance {name: item.name})
    ON CREATE SET containerInstance.status = item.status
    MERGE (docker:Docker {setup: "local"})
    MERGE (docker)-[:RUNS]->(containerInstance)
    MERGE (image:Image {name: item.image})
    MERGE (containerInstance)-[:OF]->(image)
    MERGE (cpuStats:CPUStats {cpu_usage: item.stats})
    MERGE (containerInstance)-[:USES]->(cpuStats)
    """
    neo4j_graph.query(import_query, {"data": data})

load_docker_data()