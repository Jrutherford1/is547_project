import os
import networkx as nx
from pyvis.network import Network
import json

def create_person_centric_graph(base_dir="data/Processed_Committees",
                                committee=None,
                                limit=50,
                                min_person_mentions=2):
    """
    Creates an interactive graph where clicking a person shows only their connections

    Args:
        base_dir: Base directory containing processed files
        committee: Optional committee to focus on
        limit: Maximum number of files to process
        min_person_mentions: Minimum mentions to include a person

    Returns:
        NetworkX graph object
    """
    # Build the full graph first (similar to your existing function)
    G = nx.Graph()

    # First pass: count people
    person_counts = {}
    processed = 0

    print(f"Building person-centric graph...")

    for root, _, files in os.walk(base_dir):
        if committee and committee not in root:
            continue

        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            if processed >= limit:
                break

            json_path = os.path.join(root, json_file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if "entities" in metadata and "PERSON" in metadata["entities"]:
                    for person in metadata["entities"]["PERSON"]:
                        person_counts[person] = person_counts.get(person, 0) + 1

                processed += 1

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    # Filter to important people
    important_people = {person for person, count in person_counts.items()
                        if count >= min_person_mentions}

    print(f"Found {len(important_people)} people with {min_person_mentions}+ mentions")

    # Second pass: build the graph
    processed = 0

    for root, _, files in os.walk(base_dir):
        if committee and committee not in root:
            continue

        json_files = [f for f in files if f.endswith(".json") and f != "project_metadata.jsonld"]

        for json_file in json_files:
            if processed >= limit:
                break

            json_path = os.path.join(root, json_file)
            doc_id = os.path.splitext(json_file)[0]

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if "entities" not in metadata or "PERSON" not in metadata["entities"]:
                    continue

                # Get people in this document
                people_in_doc = [p for p in metadata["entities"]["PERSON"] if p in important_people]

                if not people_in_doc:
                    continue

                # Add document node with more detailed info
                doc_title = metadata.get("name", doc_id)
                committee_name = os.path.basename(root)
                doc_type = metadata.get("type", "Unknown")
                date = metadata.get("date", "Unknown")

                # Create a shorter display title but keep full info
                display_title = doc_title[:30] + "..." if len(doc_title) > 30 else doc_title

                G.add_node(doc_id,
                           label=display_title,
                           type="document",
                           committee=committee_name,
                           full_title=doc_title,
                           doc_type=doc_type,
                           date=date,
                           size=15)

                # Add people and connect them to the document
                for person in people_in_doc:
                    person_id = f"PERSON:{person}"

                    if not G.has_node(person_id):
                        G.add_node(person_id,
                                   label=person,
                                   type="PERSON",
                                   count=person_counts[person],
                                   size=25 + min(person_counts[person] * 3, 30))

                    # Connect person to document
                    G.add_edge(doc_id, person_id, type="mentions")

                processed += 1

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def create_interactive_person_explorer(G, output_file="person_explorer.html"):
    """
    Creates an interactive visualization where clicking a person filters the view
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize")
        return

    # Create PyVis network
    # notebook=True keeps inline-friendly behavior; physics disabled to avoid animations
    net = Network(height="800px", width="100%", notebook=True)

    # Disable physics/animation entirely to prevent any layout animations or movement
    net.set_options("""
    {
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "selectConnectedEdges": true,
        "hideEdgesOnDrag": false,
        "dragNodes": false,
        "dragView": true,
        "zoomView": true
      },
      "nodes": {
        "font": {
          "size": 14
        },
        "borderWidth": 2,
        "shadow": false
      },
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "enabled": false
        },
        "shadow": false
      }
    }
    """)

    # Ensure PyVis doesn't try to toggle physics later (explicitly turn off)
    try:
        net.toggle_physics(False)
    except Exception:
        # toggle_physics may not be available in some pyvis versions; ignore safely
        pass

    # Colors for different node types
    colors = {
        "document": "#e74c3c",  # Red for documents
        "PERSON": "#3498db"     # Blue for people
    }

    # Add nodes with enhanced styling
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "unknown")

        # Set node properties based on type
        if node_type == "document":
            color = colors["document"]
            size = 20
            title = (f"📄 Document: {attrs.get('full_title', node)}<br>"
                     f"Committee: {attrs.get('committee', 'Unknown')}<br>"
                     f"Type: {attrs.get('doc_type', 'Unknown')}<br>"
                     f"Date: {attrs.get('date', 'Unknown')}")
        else:  # PERSON
            color = colors["PERSON"]
            count = attrs.get("count", 1)
            size = 30 + min(count * 2, 25)  # Larger for more mentions
            title = (f"👤 Person: {attrs.get('label', node)}<br>"
                     f"Mentioned in {count} documents<br>"
                     f"<i>Click to see only this person's connections</i>")

        # Add the node
        net.add_node(
            node,
            label=attrs.get("label", node),
            color=color,
            title=title,
            size=size,
            physics=False  # ensure each node is static
        )

    # Add edges with styling
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "mentions")

        net.add_edge(
            u, v,
            color="#95a5a6",  # Gray for all connections
            width=2,
            title=f"Connection: {edge_type}",
            smooth=False
        )

    # Add custom JavaScript for person-clicking functionality
    # Use network.fit({animation:false}) to prevent animated transitions
    custom_js = """
    <script>
    var allNodes = nodes.get();
    var allEdges = edges.get();
    var selectedPerson = null;

    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var clickedNode = params.nodes[0];
            var nodeData = nodes.get(clickedNode);

            // Check if clicked node is a person
            if (nodeData.id.startsWith('PERSON:')) {
                if (selectedPerson === clickedNode) {
                    // If same person clicked, show all nodes
                    showAllNodes();
                    selectedPerson = null;
                } else {
                    // Show only this person and their connected documents
                    showPersonOnly(clickedNode);
                    selectedPerson = clickedNode;
                }
            }
        } else {
            // Clicked on empty space, show all
            showAllNodes();
            selectedPerson = null;
        }
    });

    function showPersonOnly(personId) {
        var connectedNodes = [personId];
        var connectedEdges = [];

        // Find all edges connected to this person
        allEdges.forEach(function(edge) {
            if (edge.from === personId || edge.to === personId) {
                connectedEdges.push(edge);
                // Add the connected document
                var otherNode = edge.from === personId ? edge.to : edge.from;
                if (connectedNodes.indexOf(otherNode) === -1) {
                    connectedNodes.push(otherNode);
                }
            }
        });

        // Filter nodes and edges
        var filteredNodes = allNodes.filter(function(node) {
            return connectedNodes.indexOf(node.id) !== -1;
        });

        // Update the network without animation
        nodes.clear();
        edges.clear();
        nodes.add(filteredNodes);
        edges.add(connectedEdges);

        // Fit the view without animation
        try {
            network.fit({animation:false});
        } catch (e) {
            // fallback if fit doesn't accept options
            network.fit();
        }

        console.log("Showing person: " + personId + " with " + connectedNodes.length + " nodes");
    }

    function showAllNodes() {
        nodes.clear();
        edges.clear();
        nodes.add(allNodes);
        edges.add(allEdges);
        try {
            network.fit({animation:false});
        } catch (e) {
            network.fit();
        }
        console.log("Showing all nodes");
    }
    </script>
    """

    # Save with custom JavaScript
    net.show(output_file)

    # Add the custom JavaScript to the generated HTML
    with open(output_file, 'r') as f:
        content = f.read()

    # Insert the custom JS before the closing body tag
    content = content.replace('</body>', custom_js + '</body>')

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Interactive person explorer saved to {output_file}")
    print("Instructions:")
    print("- Click on any person (blue node) to see only their document connections")
    print("- Click the same person again or click empty space to show all nodes")
    print("- Hover over nodes to see detailed information")
    print("- Physics and animations are disabled so the graph stays static")

    return net


# Usage function that combines both
def create_person_document_explorer(base_dir="data/Processed_Committees",
                                    committee=None,
                                    limit=50,
                                    min_person_mentions=2,
                                    output_file="person_document_explorer.html"):
    """
    Complete workflow to create person-centric interactive graph
    """
    # Build the graph
    graph = create_person_centric_graph(
        base_dir=base_dir,
        committee=committee,
        limit=limit,
        min_person_mentions=min_person_mentions
    )

    # Create interactive visualization
    net = create_interactive_person_explorer(graph, output_file)

    return graph, net