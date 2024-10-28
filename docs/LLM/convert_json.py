import json
import re

def load_json(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        # Read the entire content and clean it
        content = f.read()
        cleaned_content = clean_invalid_control_characters(content)
        return json.loads(cleaned_content)

# Function to clean up invalid control characters in the JSON string
def clean_invalid_control_characters(content):
    # Remove control characters except for whitespace (like newline)
    cleaned_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\n\r]+', '', content)
    return cleaned_content


# Helper function to avoid duplicates
def add_node(node, node_list):
    if not any(n['id'] == node['properties']['id'] for n in node_list):
        node_list.append({
            "id": node['properties']['id']
            # "description": node['properties']['description']
        })

if __name__ == '__main__':
    neo4j_json = load_json("/home/weiya/Downloads/records_family.json")
    # Output data structure to store nodes and links
    nodes = []
    links = []

    # Process each relationship entry in the Neo4j JSON
    for entry in neo4j_json:
        # Add source and target nodes (avoid duplicates)
        add_node(entry["n"], nodes)
        add_node(entry["m"], nodes)
        
        # Add the link (relationship)
        links.append({
            "source": entry['n']['properties']['id'],
            "target": entry['m']['properties']['id'],
            "relationship": entry['r']["type"]
        })

    # Final output structure
    output_json = {
        "nodes": nodes,
        "links": links
    }

    # Print or save the result as a JSON file
    print(json.dumps(output_json, indent=2))

    # Optionally, write the result to a JSON file
    with open('/home/weiya/GitHub/techNotes/docs/LLM/sushi_echart.json', 'w') as f:
        json.dump(output_json, f, indent=2)
