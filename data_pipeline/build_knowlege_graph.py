import os
import json
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS


def build_knowledge_graph(base_dir="data/Processed_Committees", committee=None, limit=100):
    """
    Build a knowledge graph from JSON-LD metadata files

    Args:
        base_dir: Base directory containing processed files
        committee: Optional committee to focus on (if None, process all)
        limit: Maximum number of files to process

    Returns:
        NetworkX graph object
    """
    # Create a NetworkX graph
    G = nx.Graph()

    # Track entities and their occurrences
    entity_docs = {}  # Map entities to documents
    entity_counts = {}  # Count entity occurrences

    # Track statistics
    processed = 0

    print(f"Building knowledge graph from JSON-LD metadata...")

    # Process files
    for root, _, files in os.walk(base_dir):
        # Skip if focusing on a specific committee
        if committee and committee not in root:
            continue

        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            if processed >= limit:
                break

            json_path = os.path.join(root, json_file)
            doc_id = os.path.splitext(json_file)[0]

            try:
                # Load JSON-LD metadata
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Skip if no entities
                if "entities" not in metadata:
                    continue

                # Add document node
                doc_title = metadata.get("title", doc_id)
                G.add_node(doc_id,
                           label=doc_title,
                           type="document",
                           committee=os.path.basename(root),
                           title=doc_title)

                # Process entities
                for entity_type, entities in metadata["entities"].items():
                    for entity in entities:
                        # Create a unique ID for the entity
                        entity_id = f"{entity_type}:{entity}"

                        # Add entity node if it doesn't exist
                        if not G.has_node(entity_id):
                            G.add_node(entity_id,
                                       label=entity,
                                       type=entity_type,
                                       count=0)

                        # Track entity occurrence
                        G.nodes[entity_id]["count"] = G.nodes[entity_id].get("count", 0) + 1

                        # Connect entity to document
                        G.add_edge(doc_id, entity_id, type="mentions")

                        # Track entity-document relationships
                        if entity_id not in entity_docs:
                            entity_docs[entity_id] = []
                        entity_docs[entity_id].append(doc_id)

                        # Update counts
                        entity_counts[entity_id] = entity_counts.get(entity_id, 0) + 1

                # Connect entities that appear in the same document
                # This creates relationships between entities
                entity_ids = []
                for entity_type, entities in metadata["entities"].items():
                    for entity in entities:
                        entity_ids.append(f"{entity_type}:{entity}")

                # Connect co-occurring entities
                for i in range(len(entity_ids)):
                    for j in range(i + 1, len(entity_ids)):
                        # Add edge or increment weight if it exists
                        if G.has_edge(entity_ids[i], entity_ids[j]):
                            G[entity_ids[i]][entity_ids[j]]["weight"] += 1
                        else:
                            G.add_edge(entity_ids[i], entity_ids[j],
                                       type="co-occurs",
                                       weight=1)

                processed += 1

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    # Connect entities that appear in multiple documents
    # This creates a network of related entities
    for entity_id, docs in entity_docs.items():
        if len(docs) > 1:
            # Entity appears in multiple documents
            G.nodes[entity_id]["documents"] = len(docs)

    print(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Processed {processed} documents")

    return G


def visualize_graph(G, output_file="knowledge_graph.html", height="750px", filter_min_weight=None):
    """
    Visualize the knowledge graph using PyVis

    Args:
        G: NetworkX graph
        output_file: HTML output file
        height: Visualization height
        filter_min_weight: Minimum weight for edges (to reduce complexity)
    """
    # Create filtered graph if needed
    if filter_min_weight:
        filtered_G = nx.Graph()
        for node, attrs in G.nodes(data=True):
            filtered_G.add_node(node, **attrs)

        for u, v, attrs in G.edges(data=True):
            if attrs.get("weight", 0) >= filter_min_weight:
                filtered_G.add_edge(u, v, **attrs)

        graph_to_viz = filtered_G
    else:
        graph_to_viz = G

    # Create PyVis network
    net = Network(height=height, width="100%", notebook=True)

    # Set options
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.5,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      },
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      }
    }
    """)

    # Node colors by type
    colors = {
        "document": "#6929c4",
        "PERSON": "#1192e8",
        "ORG": "#005d5d",
        "GPE": "#9f1853",
        "DATE": "#fa4d56"
    }

    # Add nodes with appropriate colors and sizes
    for node, attrs in graph_to_viz.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        count = attrs.get("count", 1)

        # Calculate size based on importance
        size = 10
        if node_type == "document":
            size = 15
        elif count > 1:
            # Scale size by count, but not too large
            size = 10 + min(count, 10)

        # Set title (hover text)
        if node_type == "document":
            title = f"Document: {attrs.get('title', node)}<br>Committee: {attrs.get('committee', 'Unknown')}"
        else:
            title = f"{node_type}: {attrs.get('label', node)}<br>Appears in {attrs.get('documents', 1)} documents"

        # Add the node
        net.add_node(
            node,
            label=attrs.get("label", node),
            color=colors.get(node_type, "#888888"),
            title=title,
            size=size
        )

    # Add edges
    for u, v, attrs in graph_to_viz.edges(data=True):
        edge_type = attrs.get("type", "unknown")
        weight = attrs.get("weight", 1)

        # Set edge properties
        edge_props = {
            "title": f"{edge_type} (weight: {weight})",
            "value": weight
        }

        # If it's a co-occurrence, use dashed line
        if edge_type == "co-occurs":
            edge_props["dashes"] = True

        net.add_edge(u, v, **edge_props)

    # Save visualization
    net.show(output_file)
    print(f"Graph visualization saved to {output_file}")

    return net