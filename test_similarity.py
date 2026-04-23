import requests
import math
import json

url = "https://api.siliconflow.cn/v1/embeddings"
headers = {
    "Authorization": "Bearer sk-kpchnxeszrcfkgnecthwyrqfsvmredgthaobezxsnjoyjzxd",
    "Content-Type": "application/json"
}

def get_embedding(text):
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float"
    }
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()
    if "data" not in data:
        print("API error:", json.dumps(data, ensure_ascii=False))
        return None
    return data["data"][0]["embedding"]

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

pairs = [
    ("今天天气真好，我想出去散步。", "The weather is really nice today, I want to go for a walk."),
    ("人工智能正在改变世界。", "Artificial intelligence is changing the world."),
    ("这本书非常有趣，我推荐你读一读。", "This book is very interesting, I recommend you read it."),
]

vectors = {}
for i, (zh, en) in enumerate(pairs):
    v_zh = get_embedding(zh)
    v_en = get_embedding(en)
    if v_zh and v_en:
        sim = cosine_sim(v_zh, v_en)
        print(f"Pair {i+1}:")
        print(f"  CN: {zh}")
        print(f"  EN: {en}")
        print(f"  Cosine Similarity: {sim:.6f}")
        print(f"  Dimension: {len(v_zh)}")
        print()
        vectors[f"zh{i}"] = v_zh
        vectors[f"en{i}"] = v_en

print("=== Contrast: different meanings ===")
# Chinese weather vs Chinese AI
sim1 = cosine_sim(vectors["zh0"], vectors["zh1"])
print(f"  weather(CN) vs AI(CN):       {sim1:.6f}")
# Chinese weather vs English weather
sim2 = cosine_sim(vectors["zh0"], vectors["en0"])
print(f"  weather(CN) vs weather(EN):  {sim2:.6f}")
# English weather vs English AI
sim3 = cosine_sim(vectors["en0"], vectors["en1"])
print(f"  weather(EN) vs AI(EN):       {sim3:.6f}")
