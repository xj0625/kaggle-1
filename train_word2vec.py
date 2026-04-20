import pandas as pd
from gensim.models import Word2Vec

# 读取预处理后的无标签数据
df = pd.read_csv('processed_unlabeled_data.csv')

# 准备训练数据：将每个评论分词（使用空格分词）
sentences = []
for review in df['processed_review']:
    if isinstance(review, str):  # 确保是字符串
        tokens = review.split()
        if tokens:  # 确保分词后不为空
            sentences.append(tokens)

print(f"准备了 {len(sentences)} 个句子用于训练Word2Vec模型")

# 训练Word2Vec模型
model = Word2Vec(
    sentences,
    vector_size=100,  # 词向量维度
    window=5,          # 上下文窗口大小
    min_count=5,       # 最小词频
    workers=4,         # 并行处理线程数
    sg=0               # 0表示CBOW，1表示Skip-gram
)

# 保存模型
model.save('word2vec.model')

print("Word2Vec模型训练完成并保存到 word2vec.model")

# 测试模型
print("\n测试Word2Vec模型:")
if 'movie' in model.wv:
    print("与'movie'最相似的词:")
    similar_words = model.wv.most_similar('movie', topn=5)
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")

if 'good' in model.wv:
    print("\n与'good'最相似的词:")
    similar_words = model.wv.most_similar('good', topn=5)
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")

# 模型信息
print(f"\n模型词汇表大小: {len(model.wv)}")
