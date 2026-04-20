import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

"""
准备Part 3的特征：结合Tfidf和Word2Vec特征
"""

# 读取预处理后的训练数据
train_df = pd.read_csv('processed_train_data.csv')
test_df = pd.read_csv('testData.tsv', sep='\t')

# 对测试数据进行预处理
from preprocess import preprocess_text
test_df['processed_review'] = test_df['review'].apply(preprocess_text)

# 提取Tfidf特征
print("提取Tfidf特征...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_review'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_review'])

# 转换为密集矩阵
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

print(f"Tfidf特征维度: {X_train_tfidf.shape[1]}")

# 加载Word2Vec特征
print("\n加载Word2Vec特征...")
X_train_word2vec = np.load('X_train_word2vec.npy')
X_test_word2vec = np.load('X_test_word2vec.npy')

print(f"Word2Vec特征维度: {X_train_word2vec.shape[1]}")

# 组合特征
print("\n组合特征...")
X_train_combined = np.hstack((X_train_tfidf, X_train_word2vec))
X_test_combined = np.hstack((X_test_tfidf, X_test_word2vec))

print(f"组合特征维度: {X_train_combined.shape[1]}")
print(f"训练数据数量: {X_train_combined.shape[0]}")
print(f"测试数据数量: {X_test_combined.shape[0]}")

# 保存组合特征
np.save('X_train_combined.npy', X_train_combined)
np.save('X_test_combined.npy', X_test_combined)

print("\n组合特征已保存到:")
print("- X_train_combined.npy")
print("- X_test_combined.npy")
