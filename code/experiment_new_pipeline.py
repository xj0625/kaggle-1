import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

"""
新实验线路：清洗+word2vec+求均值embedding+逻辑回归
"""

# 1. 文本清洗函数
def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 转换为小写
    text = text.lower()
    # 处理标点符号，保留情感相关的标点
    text = re.sub(r'[\!\?\.\,\;\:\-]', ' ', text)
    # 处理缩写
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'s", ' is', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'m", ' am', text)
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. 分词函数
def tokenize_text(text):
    try:
        # 简单的分词方法，避免nltk依赖问题
        # 按空格分词
        tokens = text.split()
        # 过滤停用词，保留否定词
        stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                        'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                        'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
                        'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
        # 保留否定词
        negation_words = set(['not', 'no', 'never', 'nor'])
        filtered_tokens = [token for token in tokens if token not in stopwords or token in negation_words]
        return filtered_tokens
    except Exception as e:
        print(f"分词错误: {e}")
        return []

# 3. 加载并预处理训练数据
print("加载并预处理训练数据...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t')
train_df['cleaned_review'] = train_df['review'].apply(clean_text)
train_df['tokens'] = train_df['cleaned_review'].apply(tokenize_text)

# 4. 加载并预处理无标签数据
print("\n加载并预处理无标签数据...")
unlabeled_df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', on_bad_lines='warn')
unlabeled_df = unlabeled_df.head(49998)  # 取49998条数据
unlabeled_df['cleaned_review'] = unlabeled_df['review'].apply(clean_text)
unlabeled_df['tokens'] = unlabeled_df['cleaned_review'].apply(tokenize_text)

# 5. 合并所有文本用于Word2Vec训练
all_tokens = train_df['tokens'].tolist() + unlabeled_df['tokens'].tolist()

# 过滤空列表
all_tokens = [tokens for tokens in all_tokens if len(tokens) > 0]
print(f"过滤后有效文本数量: {len(all_tokens)}")

# 6. 训练Word2Vec模型
print("\n训练Word2Vec模型...")
# 使用更明确的词汇表构建和训练步骤
word2vec_model = Word2Vec(
    vector_size=100,
    window=5,
    min_count=5,
    workers=4
)

# 构建词汇表
word2vec_model.build_vocab(all_tokens)
print(f"词汇表大小: {len(word2vec_model.wv)}")

# 训练模型
word2vec_model.train(all_tokens, total_examples=word2vec_model.corpus_count, epochs=10)

# 保存Word2Vec模型
word2vec_model.save('word2vec_new.model')
print(f"Word2Vec模型已保存，词汇表大小: {len(word2vec_model.wv)}")

# 7. 提取均值embedding
def get_embedding(tokens, model, vector_size=100):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

print("\n提取训练数据的均值embedding...")
X_train = np.array([get_embedding(tokens, word2vec_model) for tokens in train_df['tokens']])
y_train = train_df['sentiment'].values

# 8. 分割训练集和验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 9. 训练逻辑回归模型
print("\n训练逻辑回归模型...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_split, y_train_split)

# 10. 评估模型
print("\n评估模型性能...")
y_pred = clf.predict(X_val)
y_pred_proba = clf.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

print(f"模型准确率: {accuracy:.4f}")
print(f"模型AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_val, y_pred))

# 11. 保存模型
joblib.dump(clf, 'word2vec_logreg_new.pkl')
print("\n模型已保存到 word2vec_logreg_new.pkl")

# 12. 处理测试数据
print("\n处理测试数据...")
test_df = pd.read_csv('testData.tsv', sep='\t')
test_df['cleaned_review'] = test_df['review'].apply(clean_text)
test_df['tokens'] = test_df['cleaned_review'].apply(tokenize_text)

X_test = np.array([get_embedding(tokens, word2vec_model) for tokens in test_df['tokens']])

# 13. 生成预测结果
y_test_pred = clf.predict(X_test)

# 14. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': y_test_pred
})

submission_df.to_csv('word2vec_logreg_new_submission.csv', index=False)
print("\n提交文件已生成到 word2vec_logreg_new_submission.csv")

print("\n新实验线路执行完成！")
