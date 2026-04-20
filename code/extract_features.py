import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# 加载Word2Vec模型
model = Word2Vec.load('word2vec.model')
vector_size = model.vector_size

# 函数：将文本转换为词向量的平均值
def text_to_vector(text):
    if not isinstance(text, str):
        return np.zeros(vector_size)
    
    words = text.split()
    word_vectors = []
    
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    if not word_vectors:
        return np.zeros(vector_size)
    
    return np.mean(word_vectors, axis=0)

# 处理训练数据
train_df = pd.read_csv('processed_train_data.csv')
train_vectors = []

for review in train_df['processed_review']:
    vector = text_to_vector(review)
    train_vectors.append(vector)

# 保存训练数据的特征
X_train = np.array(train_vectors)
y_train = train_df['sentiment'].values

np.save('X_train_word2vec.npy', X_train)
np.save('y_train.npy', y_train)

print(f"训练数据特征提取完成，共 {len(X_train)} 条数据，特征维度为 {X_train.shape[1]}")

# 处理测试数据
test_df = pd.read_csv('testData.tsv', sep='\t')
from preprocess import preprocess_text

test_df['processed_review'] = test_df['review'].apply(preprocess_text)
test_vectors = []

for review in test_df['processed_review']:
    vector = text_to_vector(review)
    test_vectors.append(vector)

# 保存测试数据的特征
X_test = np.array(test_vectors)
np.save('X_test_word2vec.npy', X_test)

print(f"测试数据特征提取完成，共 {len(X_test)} 条数据，特征维度为 {X_test.shape[1]}")
