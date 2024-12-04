import os
from tqdm import tqdm
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_model_complete, hf_embedding
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# WORKING_DIR = "./dickens"
WORKING_DIR = "./CPC_PCIE"
if os.path.exists(WORKING_DIR) is False:
    os.makedirs(WORKING_DIR, exist_ok=True)
####
LLM_MODEL_NAME = '/home/chizhang/temp_folder/Qwen2.5-14B-Instruct'
# EMBED_MODEL_NAME = '/home/chizhang/temp_folder/all-MiniLM-L6-v2'
EMBED_MODEL_NAME = '/home/chizhang/temp_folder/all-MiniLM-L12-v2'

rag = LightRAG(
    working_dir=WORKING_DIR,
    chunk_token_size=10000,
    chunk_overlap_token_size=1000,
    entity_summary_to_max_tokens=2500,
    llm_model_func=hf_model_complete,
    llm_model_name=LLM_MODEL_NAME,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                EMBED_MODEL_NAME
            ),
            embed_model=AutoModel.from_pretrained(
                EMBED_MODEL_NAME
            ),
        ),
    ),
)


# with open("/home/chizhang/JIRA_Monitor_new/data/AQ/processed/AQ-7989.txt") as f:
#     rag.insert(f.read())


_folder = '/home/chizhang/JIRA_Monitor_new/data/CPC_BUG_PAE/processed'
_target_files= []
for file in os.listdir(_folder):
    _target_files.append(file)
# _target_files = _target_files[-1000:]
for file in tqdm(_target_files, desc='Lighting RAG...'):
    with open(os.path.join(_folder, file)) as f:
        rag.insert(f.read())

# Perform naive search
# print(
#     rag.query("What is this issue about?", param=QueryParam(mode="naive",
#                                                             max_token_for_text_unit=10*1024,
#                                                             max_token_for_global_context=10*1024,
#                                                             max_token_for_local_context=10*1024
#                                                             ))
# )
query = 'What is the failure about?'
print('NAIVE:')
print(
    rag.query(query, param=QueryParam(mode="naive"))
)

# Perform local search
print('local:')
print(
    rag.query(query, param=QueryParam(mode="local"))
)

# # Perform global search
print('global:')
print(
    rag.query(query, param=QueryParam(mode="global",
                                                            max_token_for_text_unit=10*1024,
                                                            max_token_for_global_context=10*1024,
                                                            max_token_for_local_context=10*1024
                                                            ))
)

# # Perform hybrid search
print('hybrid:')
print(
    rag.query(query, param=QueryParam(mode="hybrid",
                                                            max_token_for_text_unit=10*1024,
                                                            max_token_for_global_context=10*1024,
                                                            max_token_for_local_context=10*1024
                                                            ))
)
