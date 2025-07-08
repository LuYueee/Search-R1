export save_path=/home/jovyan/work_vol90/RL+RAG/Search-R1-main/indexing_corpus
export index_file=$save_path/e5_Flat.index
export corpus_file=$save_path/wiki-18.jsonl
export retriever_name=e5
#export retriever_path=intfloat/e5-base-v2
export retriever_path="/home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/intfloat_e5-base-v2"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu
