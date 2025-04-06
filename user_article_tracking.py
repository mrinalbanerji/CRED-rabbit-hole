import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configurable Paths ---
BASE_DIR = "/workspace"
ARTICLE_FOLDER = os.path.join(BASE_DIR, "wikipedia_articles_50_per_topic")
ARTICLE_METADATA_PATH = os.path.join(BASE_DIR, "article_metadata.json")
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, "user_article_knowledge_graph.graphml")
USER_PROFILE_DIR = os.path.join(BASE_DIR, "employee-profiles")

# --- Lazy Cache ---
_article_cache = None
_user_cache = None

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utility: Hashing for change detection ---
def compute_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# --- Utility: Infer topic from folder structure ---
def infer_topic_from_path(path):
    parts = Path(path).parts
    for part in parts:
        if part.lower() in ["finance", "datascience", "marketing", "sales", "artificial intelligence"]:
            return part.capitalize()
    return "General"

def update_article_metadata():
    global _article_cache

    if os.path.exists(ARTICLE_METADATA_PATH):
        with open(ARTICLE_METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    updated_articles = []
    for root, _, files in os.walk(ARTICLE_FOLDER):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ARTICLE_FOLDER)
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                hash_val = compute_hash(content)
                mod_time = os.path.getmtime(full_path)
                mod_iso = datetime.fromtimestamp(mod_time).isoformat(timespec="minutes")
                topic = infer_topic_from_path(rel_path)
                if rel_path not in metadata or metadata[rel_path]["hash"] != hash_val:
                    metadata[rel_path] = {
                        "topic": topic,
                        "last_updated": mod_iso,
                        "hash": hash_val
                    }
                    updated_articles.append(rel_path)

    with open(ARTICLE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    _article_cache = metadata
    return updated_articles

# --- Step 2: Build or update knowledge graph ---
def build_knowledge_graph(users):
    global _article_cache

    if os.path.exists(KNOWLEDGE_GRAPH_PATH):
        G = nx.read_graphml(KNOWLEDGE_GRAPH_PATH)
    else:
        G = nx.Graph()

    if _article_cache is None:
        with open(ARTICLE_METADATA_PATH, "r") as f:
            _article_cache = json.load(f)

    articles = _article_cache
    article_keys = list(articles.keys())
    article_contents = []
    for art in article_keys:
        full_path = os.path.join(ARTICLE_FOLDER, art)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                article_contents.append(f.read())
        except FileNotFoundError:
            article_contents.append(art)  # fallback
    article_embeddings = embedder.encode(article_contents, convert_to_numpy=True, normalize_embeddings=True)

    for user in users:
        uid = user["id"]
        G.add_node(uid, type="user")
        profile_text = user["Department"] + " " + " ".join(user["Interests"])
        user_emb = embedder.encode([profile_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = cosine_similarity([user_emb], article_embeddings)[0]
        top_indices = sims.argsort()[::-1][:10]
        for idx in top_indices:
            art_path = article_keys[idx]
            G.add_node(art_path, type="article")
            G.add_edge(uid, art_path, type="semantic_link", score=float(sims[idx]))

    nx.write_graphml(G, KNOWLEDGE_GRAPH_PATH)
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx(G, pos, with_labels=True, font_size=8, node_size=500, node_color='lightblue', edge_color='gray')
    plt.title("User-Article Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "knowledge_graph_viz.png"))
    return G

# --- Step 3: Save the graph explicitly (optional cleanup or session end) ---
def save_knowledge_graph(graph):
    nx.write_graphml(graph, KNOWLEDGE_GRAPH_PATH)

# --- Step 4: User update checking ---
def check_for_user_updates(user):
    global _article_cache

    user_id = user["id"]
    user_file_path = os.path.join(USER_PROFILE_DIR, f"{user_id}.json")

    if not os.path.exists(KNOWLEDGE_GRAPH_PATH):
        raise FileNotFoundError("Knowledge graph not found. Run build_knowledge_graph() first.")

    G = nx.read_graphml(KNOWLEDGE_GRAPH_PATH)

    if _article_cache is None:
        with open(ARTICLE_METADATA_PATH, "r") as f:
            _article_cache = json.load(f)

    articles = _article_cache

    if "last_notified" not in user:
        user["last_notified"] = "1970-01-01T00:00:00"

    last_notified = datetime.fromisoformat(user["last_notified"])
    related_articles = set()

    for neighbor in G.neighbors(user_id):
        if G.nodes[neighbor].get("type") == "article":
            updated_at = articles.get(neighbor, {}).get("last_updated")
            if updated_at:
                updated_time = datetime.fromisoformat(updated_at)
                if updated_time > last_notified:
                    related_articles.add(neighbor)

    user["last_notified"] = datetime.now().isoformat(timespec="minutes")
    os.makedirs(USER_PROFILE_DIR, exist_ok=True)
    with open(user_file_path, "w") as f:
        json.dump(user, f, indent=2)

    if related_articles:
        topic_counts = {}
        for article in related_articles:
            topic = articles[article]["topic"]
            topic_counts.setdefault(topic, []).append(article)

        flat_list = [f"{a} ({t})" for t, arts in topic_counts.items() for a in arts]
        return f"Since your last visit, the following articles were updated: {', '.join(flat_list)}"
    else:
        return "No new article updates since your last interaction."

# --- Optional: Load users from JSON files in a folder ---
def load_users_from_json(folder):
    global _user_cache

    if _user_cache is not None:
        return _user_cache

    users = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
                if "id" not in data:
                    data["id"] = file.split(".")[0]
                users.append(data)

    _user_cache = users
    return users

# --- Initialization for reuse ---
def initialize_graph_and_check_updates(user_dir=USER_PROFILE_DIR):
    users = load_users_from_json(user_dir)
    update_article_metadata()
    G = build_knowledge_graph(users)
    print("Graph Initialized")

def get_update_message(KNOWLEDGE_GRAPH_PATH, users):
    G = nx.read_graphml(KNOWLEDGE_GRAPH_PATH)
    update_article_metadata()
    update_messages = {}
    for user in users:
        update_messages[user["id"]] = check_for_user_updates(user)
    save_knowledge_graph(G)
    return update_messages
