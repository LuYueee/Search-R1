
## Search Engine

In this document, we provide examples of how to launch different retrievers, including local sparse retriever (e.g., BM25), local dense retriever (e.g., e5) and online search engine.
For local retrievers, we use [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) corpus as an example and the corpus indexing can be found at [bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index), [e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index), [e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64).

### How to choose the retriever?

- If you have a private or domain-specific corpus, choose **local retriever**.

    - If there is no high quality embedding-based retrievers (dense retrievers) in your domain, choose **sparse local retriever** (e.g., BM25).

    - Otherwise choose **dense local retriever**.
    
        - If you do not have sufficent GPUs to conduct exact dense embedding matching, choose **ANN indexing** on CPUs.

        - If you have sufficient GPUs, choose **flat indexing** on GPUs.


- If you want to train a general LLM search agent and have enough funding, choose **online search engine** (e.g., [SerpAPI](https://serpapi.com/)).


- If you have a domain specific online search engine (e.g., PubMed search), you can refer to [link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py) to integrate it to Search-R1 by yourself.

Search engine launching scripts can be found at [link](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever).

### Local Sparse Retriever

Sparse retriever (e.g., bm25) is a traditional method. The retrieval process is very efficient and no GPUs are needed. However, it may not be as accurate as dense retrievers in some specific domain.

(1) Download the indexing.
```bash
save_path=/your/path/to/save
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

(2) Launch a local BM25 retriever server.
```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```


### Local Dense Retriever

You can also adopt some off-the-shelf dense retrievers, e.g., e5. These models are much stronger than sparse retriever in some specific domains.
If you have sufficient GPU, we would recommend the flat indexing variant below, otherwise you can adopt the ANN variant.

#### Flat indexing

Flat indexing conducts exact embedding match, which is slow but very accurate. To make it efficient enough to support online RL, we would recommend enable **GPU** usage by ```--faiss_gpu```.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```


####################################################################
(2) Launch a local flat e5 retriever server.
####################################################################
```bash
conda activate searchr1
pip3 install nvidia-cublas-cu12==12.3.4.1 
######################################################################
sudo sh cuda_12.1.0_530.30.02_linux.run
# 临时生效
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# 永久生效（推荐写入 ~/.bashrc 或 /etc/profile）
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
#################################################################################
conda create -n retriever python=3.10
conda activate retriever
# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0=cuda120_py310h2c91c31_301 \
             torchvision==0.19.0=cuda120py310h8d5198f_0 \
             -c conda-forge
pip install transformers datasets pyserini
## API function
pip install uvicorn fastapi
pip install -r requirements.txt

## install the gpu version faiss to guarantee efficient RL rollout
pip install /home/jovyan/work_vol90/RL+RAG/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


############################################################3
pip install faiss-gpu
conda install faiss-gpu-1.8.0-h0240f8b_2.conda 
# CPU版本和GPU版本的faiss只能装一个
# CPU版本：conda install -c conda-forge faiss=1.8.0=py310cuda120h3ec4162_1_cuda
pip install torchaudio==2.4.0 
conda install pytorch==2.4.0=cuda120_py310h2c91c31_301 \
             torchvision==0.19.0=cuda120py310h8d5198f_0 \
             -c conda-forge

######################################################################
启动retriever
######################################################################
conda activate retriever
export save_path=/home/jovyan/work_vol90/RL+RAG/Search-R1-main/indexing_corpus
export index_file=$save_path/e5_Flat.index
export corpus_file=$save_path/wiki-18.jsonl
export retriever_name=e5
#export retriever_path=intfloat/e5-base-v2
export retriever_path="/home/jovyan/work_vol90/RL+RAG/Search-R1-main/models/intfloat_e5-base-v2"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu



####################################################################
关闭retriever
#####################################################################
# 查找进程ID
pgrep -f "retrieval_server.py" | xargs kill -9

# 确认进程是否终止（若无输出则已关闭）
ps aux | grep "retrieval_server.py"

######################################################################
测试retriever
######################################################################

基于你的描述，你现在已经成功启动了检索服务器。当你看到 "Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)" 时，这意味着检索服务器正在运行并监听8000端口。

关于终端使用
你有两个选择：

保持当前终端运行：检索服务器需要持续运行才能响应请求，所以这个终端需要保持打开状态
使用新终端进行测试：打开一个新的终端窗口来测试检索器功能
如何测试检索器
代码库提供了几种测试检索器的方法：

方法1：使用提供的测试脚本
retrieval_request.py:1-25
这个脚本展示了如何向检索服务器发送请求。你可以运行：

python search_r1/search/retrieval_request.py

方法2：在推理脚本中测试
infer.py:61-79
推理脚本中的 search() 函数展示了如何调用检索服务器。当你运行推理时，模型会自动调用这个函数来搜索相关信息。

方法3：直接HTTP请求测试
你也可以使用curl命令直接测试：

curl -X POST "http://127.0.0.1:8000/retrieve" \
-H "Content-Type: application/json" \
-d '{"queries": [ "What is Python?"], "topk": 3, "return_scores": true}'

```


#### ANN indexing (HNSW64)

To improve the search efficient with only **CPU**, you can adopt approximate nearest neighbor (ANN) indexing, e.g., with HNSW64.
It is very efficient, but may not be as accurate as flat indexing, especially when the number of retrieved passages is small.

(1) Download the indexing.
```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```


(2) Launch a local ANN dense retriever server.
```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```


### Online Search Engine

We support both [Google Search API](https://developers.google.com/custom-search/v1/overview) and [SerpAPI](https://serpapi.com/). We would recommend [SerpAPI](https://serpapi.com/) since it integrates multiple online search engine APIs (including Google, Bing, Baidu, etc) and does not have a monthly quota limitation ([Google Search API](https://developers.google.com/custom-search/v1/overview) has a hard 10k monthly quota, which is not sufficient to fulfill online LLM RL training).

#### SerAPI online search server

```bash
search_url=https://serpapi.com/search
serp_api_key="" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google online search server

```bash
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```

