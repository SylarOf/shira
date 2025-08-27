import asyncio
import json
import re
from llm.prompts import PROMPTS
from graph import NXGraph
from database import VectorDB
from collections import defaultdict

class KG:
    
    def __init__(self,llm):
        self.graph = NXGraph
        self.db = VectorDB
        self.llm = llm 
    async def map(self,chunk = str):
        ""
        
    def reduce(self, results):
        ""

    def query(self,query = str):
        ""

    def extract_entities_from_file(self,file):
        ""


async def extract_entities_from_text(
    text: str,
    graph_db: NXGraph,
    entity_vdb: VectorDB = None,
    use_llm_func=None,
    max_iterations: int = 2,
):
    
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    async def _process_chunk(chunk: str):
        # Step 1: 初次抽取实体+关系
        prompt = f"Extract entities and relationships from the text:\n{chunk}\nOutput as JSON: {{'entities':[{{'name':'','type':'','description':''}}], 'relations':[{{'src':'','tgt':'','relation':'','description':''}}]}}"
        result = await use_llm_func(prompt)
        try:
            data = json.loads(result)
        except Exception:
            data = {"entities": [], "relations": []}

        # Step 2: 多轮迭代抽取遗漏实体
        for _ in range(max_iterations - 1):
            prompt_iter = f"Continue extracting entities and relationships from the previous output:\n{json.dumps(data)}\nText: {chunk}\nOutput as JSON with new entities/relations only"
            iter_result = await use_llm_func(prompt_iter)
            try:
                iter_data = json.loads(iter_result)
                data["entities"].extend(iter_data.get("entities", []))
                data["relations"].extend(iter_data.get("relations", []))
            except Exception:
                pass

        return data

    # 异步并行处理所有 chunk
    tasks = [_process_chunk(c) for c in chunks]
    results = await asyncio.gather(*tasks)

    # 合并结果
    all_entities = []
    all_relations = []
    for r in results:
        all_entities.extend(r.get("entities", []))
        all_relations.extend(r.get("relations", []))

    # 去重实体
    nodes_map = {}
    for ent in all_entities:
        ent_id = compute_md5_id(ent["name"], prefix="ent-")
        nodes_map[ent_id] = ent

    # 去重关系
    edges_map = {}
    for rel in all_relations:
        edge_id = compute_md5_id(rel["src"] + rel["tgt"] + rel["relation"], prefix="rel-")
        edges_map[edge_id] = rel

    # 插入图数据库
    for node_id, node in nodes_map.items():
        await graph_db.upsert_node(node_id, node)

    for edge_id, edge in edges_map.items():
        await graph_db.upsert_edge(edge["src"], edge["tgt"], edge)

    # 插入向量数据库
    if entity_vdb:
        entity_data = {
            compute_md5_id(ent["name"], prefix="ent-"): {
                "content": ent["name"] + ent.get("description", ""),
                "entity_name": ent["name"],
            } for ent in nodes_map.values()
        }
        await entity_vdb.upsert(entity_data)

    if relation_vdb:
        relation_data = {
            compute_md5_id(rel["src"] + rel["tgt"], prefix="rel-"): {
                "src_id": rel["src"],
                "tgt_id": rel["tgt"],
                "content": rel.get("relation", "") + rel["src"] + rel["tgt"] + rel.get("description", "")
            } for rel in edges_map.values()
        }
        await relation_vdb.upsert(relation_data)

    return nodes_map, edges_map


async def extract_keywords_from_query(query,graph,vdb):
    keywords_prompt = PROMPTS[keywords_prompt]

    result = await use_model_func(keywords_prompt)


async def extract_entities(chunk:str,graph,entity_vdb,relationships_vdb,config):
    use_llm_func = PROMPTS['llm_model_func']
    lanuage = PROMPTS['DEFAULT_LANGUAGE']
    entity_types = PROMPTS['DEFAULT_ENTITY_TYPES']

    examples = "\n".join(PROMPTS['entity_extraction_examples'])

    example_context_base = dict(
        tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter = PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
        entity_types = ",".join(entity_types),
        lanuage = lanuage,
    )

    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS['entity_extraction']

    context_base = dict(
        tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter = PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
        entity_types = ",".join(entity_types),
        examples = examples,
        lanuage = lanuage,
    )

    async def _process_single_content(chunk:str):
        content = chunk

        hint_prompt = entity_extract_prompt.format(**context_base,input_text = content)

        result = await use_llm_func(hint_prompt)

        records = split_string_by_multi_markers(
            result,[context_base["record_delimiter"],context_base["completion_delimiter"]]
        ) 


        for record in records:
            record = re.search(r'\((.*)\)')
            if record is None:
                continue

            record = split_string_by_multi_markers(record,[context_base['tuple_delimiter']])

            entities = "" 
            relations = ""
        
            
    




def split_string_by_multi_markers():
    ""