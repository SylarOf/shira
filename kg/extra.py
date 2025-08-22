import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. 定义示例文档 ---
# 我们假设有 k=4 个与某个项目相关的文档块
documents = [
    # 文档 1: 介绍公司和创始人
    "Innovatech Solutions was founded in 2021 by computer scientist Dr. Evelyn Reed. The company is headquartered in Silicon Valley.",
    # 文档 2: 介绍公司的主要产品
    "The flagship product of Innovatech Solutions is 'QuantumLeap', a new generation AI processor. This processor was designed by a team led by Dr. Reed.",
    # 文档 3: 介绍产品的应用
    "QuantumLeap is primarily used for advanced data analysis and has been adopted by major tech firms. It significantly speeds up machine learning tasks.",
    # 文档 4: 提到创始人的背景和另一个项目
    "Dr. Evelyn Reed previously worked at a research institute where she developed a project named 'Synapse'. The work on 'Synapse' laid the groundwork for the QuantumLeap processor."
]


def extract_svo(doc):
    """
    提取主谓宾(Subject-Verb-Object)三元组作为基本关系
    这是一个简化的关系提取方法，真实场景可能需要更复杂的模型
    """
    triples = []
    for sent in doc.sents:
        # 寻找动词及其主语和宾语
        for token in sent:
            if "VERB" in token.pos_:
                subject = None
                obj = None
                for child in token.children:
                    if "subj" in child.dep_: # nsubj, csubj
                        subject = child
                    if "obj" in child.dep_: # dobj, pobj
                        obj = child

                if subject and obj:
                    # 规范化文本，只取原形
                    subject_text = ' '.join(t.lemma_.lower() for t in subject.subtree).strip()
                    verb_text = token.lemma_.lower()
                    object_text = ' '.join(t.lemma_.lower() for t in obj.subtree).strip()
                    
                    # 过滤掉代词
                    if subject.lemma_ != '-PRON-' and obj.lemma_ != '-PRON-':
                        triples.append((subject_text, verb_text, object_text))
    return triples


def build_knowledge_graph(docs):
    """
    从一系列文档中构建知识图谱
    """
    # 加载 spaCy 模型
    nlp = spacy.load("en_core_web_lg")
    
    # 初始化一个有向图
    G = nx.DiGraph()

    print("--- 开始构建知识图谱 ---")
    for i, doc_text in enumerate(docs):
        print(f"处理文档 {i+1}/{len(docs)}...")
        doc = nlp(doc_text)
        
        # --- 步骤 2.1: 提取实体并添加为节点 ---
        for ent in doc.ents:
            # 节点名称使用实体的原形并转为小写，实现归一化
            node_name = ent.lemma_.lower().strip()
            if not G.has_node(node_name):
                G.add_node(node_name, label=ent.label_, sources=[f"Doc_{i+1}"])
            else:
                # 如果节点已存在，追加来源信息
                G.nodes[node_name]['sources'].append(f"Doc_{i+1}")

        # --- 步骤 2.2: 提取关系并添加为边 ---
        # 我们使用简化的SVO提取法
        svo_triples = extract_svo(doc)
        for subj, verb, obj in svo_triples:
            # 确保主语和宾语都是图中的节点
            if G.has_node(subj) and G.has_node(obj):
                # 添加从主语到宾语的有向边，边的标签是动词
                G.add_edge(subj, obj, label=verb)

    print("--- 知识图谱构建完成 ---")
    print(f"图中有 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    return G

def build_semantic_index(graph, model):
    """
    为图中的每个节点创建语义嵌入索引
    """
    print("\n--- 开始构建语义索引 ---")
    node_ids = list(graph.nodes())
    
    # 我们直接使用节点名称作为嵌入的文本内容
    node_texts = [node_id for node_id in node_ids]
    
    embeddings = model.encode(node_texts, show_progress_bar=True)
    
    semantic_index = {
        "node_ids": node_ids,
        "embeddings": embeddings
    }
    print("--- 语义索引构建完成 ---")
    return semantic_index

def find_relevant_nodes(query, model, index, top_k=3):
    """
    4.1 NodeRetrieval 的实现: 根据查询找到最相关的节点
    """
    print(f"\n--- 正在为查询 '{query}' 检索相关节点 ---")
    query_embedding = model.encode([query])
    
    # 计算查询与所有节点嵌入的余弦相似度
    sims = cosine_similarity(query_embedding, index["embeddings"])[0]
    
    # 获取最相似的 top_k 个节点的索引
    top_k_indices = np.argsort(sims)[-top_k:][::-1]
    
    # 返回最相关的节点ID和它们的相似度分数
    relevant_nodes = [(index["node_ids"][i], sims[i]) for i in top_k_indices]
    return relevant_nodes

def visualize_graph(G):
    """
    使用 Matplotlib 将图可视化
    """
    plt.figure(figsize=(16, 12))
    
    pos = nx.spring_layout(G, k=0.9, iterations=50) # 使用 spring 布局
    
    # 绘制节点和标签
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # 绘制边和边的标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    plt.title("构建的知识图谱", size=20)
    plt.show()


if __name__ == '__main__':
    # --- 主流程 ---
    
    # 1. 从文档构建知识图谱
    knowledge_graph = build_knowledge_graph(documents)
    
    # 2. 加载语义嵌入模型
    # all-MiniLM-L6-v2 是一个轻量且高效的模型
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. 为图节点构建语义索引
    semantic_index = build_semantic_index(knowledge_graph, embedding_model)
    
    # 4. 模拟 "4.1 NodeRetrieval" 阶段
    query = "Who is the founder of the company that created QuantumLeap?"
    retrieved_nodes = find_relevant_nodes(query, embedding_model, semantic_index, top_k=3)
    
    print("\n--- 检索结果 ---")
    print(f"与查询 '{query}' 最相关的节点是:")
    for node, score in retrieved_nodes:
        print(f"- 节点: '{node}' (相似度: {score:.4f})")
        print(f"  (来源: {knowledge_graph.nodes[node].get('sources', 'N/A')})")

    # 5. 可视化图谱
    visualize_graph(knowledge_graph)