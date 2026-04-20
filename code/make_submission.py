import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import sparse
import joblib
import os

os.makedirs('artifacts', exist_ok=True)

def get_tfidf_features(train_texts, test_texts, vectorizer_type='word', max_features=5000, ngram_range=(1,2)):
    if vectorizer_type == 'word':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\w{1,}'
        )
    elif vectorizer_type == 'char':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            analyzer='char',
            strip_accents='unicode'
        )
    elif vectorizer_type == 'both':
        vectorizer_word = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\w{1,}'
        )
        vectorizer_char = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            analyzer='char',
            strip_accents='unicode'
        )
        X_train_word = vectorizer_word.fit_transform(train_texts)
        X_test_word = vectorizer_word.transform(test_texts)
        X_train_char = vectorizer_char.fit_transform(train_texts)
        X_test_char = vectorizer_char.transform(test_texts)
        X_train = sparse.hstack([X_train_word, X_train_char]).tocsr()
        X_test = sparse.hstack([X_test_word, X_test_char]).tocsr()
        return X_train, X_test, (vectorizer_word, vectorizer_char)
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

class NBSVM:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.r = None
        self.clf = None
    
    def fit(self, X, y):
        pos_mask = y == 1
        neg_mask = y == 0
        
        p = self.alpha + np.sum(X[pos_mask], axis=0)
        q = self.alpha + np.sum(X[neg_mask], axis=0)
        
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        
        self.r = np.log((p / p.sum()) / (q / q.sum()))
        
        X_new = X.multiply(self.r)
        self.clf = LogisticRegression(max_iter=1000, C=4.0, solver='liblinear')
        self.clf.fit(X_new, y)
        return self
    
    def predict_proba(self, X):
        X_new = X.multiply(self.r)
        return self.clf.predict_proba(X_new)

def train_tfidf_nbsvm():
    print("加载清洗后的数据...")
    train_df = pd.read_csv('train_cleaned.csv')
    test_df = pd.read_csv('test_cleaned.csv')
    
    train_texts = train_df['cleaned_review'].values
    test_texts = test_df['cleaned_review'].values
    y_train = train_df['sentiment'].values
    
    print(f"训练样本: {len(train_texts)}, 测试样本: {len(test_texts)}")
    
    print("\n提取TF-IDF特征（word+char n-grams）...")
    X_train, X_test, vectorizers = get_tfidf_features(
        train_texts, test_texts, 
        vectorizer_type='both',
        max_features=30000,
        ngram_range=(1,3)
    )
    
    print(f"特征维度: {X_train.shape}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print("\n训练NBSVM模型...")
    nbsvm = NBSVM(alpha=1.0)
    nbsvm.fit(X_tr, y_tr)
    
    y_val_pred = nbsvm.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"验证集AUC: {val_auc:.4f}")
    
    print("\n在全部训练数据上重新训练...")
    nbsvm_full = NBSVM(alpha=1.0)
    nbsvm_full.fit(X_train, y_train)
    
    print("保存模型...")
    joblib.dump(nbsvm_full, 'artifacts/nbsvm_both_model.pkl')
    joblib.dump(vectorizers[0], 'artifacts/tfidf_word_vectorizer.pkl')
    joblib.dump(vectorizers[1], 'artifacts/tfidf_char_vectorizer.pkl')
    
    print("\n生成预测...")
    y_test_pred = nbsvm_full.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_test_pred
    })
    submission.to_csv('submission_tfidf_nbsvm.csv', index=False)
    print(f"提交文件已保存: submission_tfidf_nbsvm.csv")
    
    return val_auc

def train_word2vec_features():
    print("\n" + "="*50)
    print("训练Word2Vec模型...")
    from gensim.models import Word2Vec
    
    train_df = pd.read_csv('train_cleaned.csv')
    unlabeled_df = pd.read_csv('unlabeled_cleaned.csv')
    test_df = pd.read_csv('test_cleaned.csv')
    
    def tokenize(text):
        return text.split()
    
    train_tokens = [tokenize(text) for text in train_df['cleaned_review']]
    unlabeled_tokens = [tokenize(text) for text in unlabeled_df['cleaned_review']]
    test_tokens = [tokenize(text) for text in test_df['cleaned_review']]
    
    all_tokens = train_tokens + unlabeled_tokens
    all_tokens = [t for t in all_tokens if len(t) > 0]
    
    print(f"训练Word2Vec，语料数量: {len(all_tokens)}")
    w2v = Word2Vec(
        sentences=all_tokens,
        vector_size=200,
        window=5,
        min_count=3,
        workers=4,
        epochs=15,
        sg=1
    )
    w2v.save('artifacts/word2vec.model')
    print(f"Word2Vec词汇表大小: {len(w2v.wv)}")
    
    def get_avg_embedding(tokens, model, dim=200):
        vectors = []
        for t in tokens:
            if t in model.wv:
                vectors.append(model.wv[t])
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(dim)
    
    print("提取Word2Vec特征...")
    X_train_w2v = np.array([get_avg_embedding(t, w2v) for t in train_tokens])
    X_test_w2v = np.array([get_avg_embedding(t, w2v) for t in test_tokens])
    
    y_train = train_df['sentiment'].values
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_w2v, y_train, test_size=0.2, random_state=42)
    
    print("训练逻辑回归...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    
    y_val_pred = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"Word2Vec + LR 验证集AUC: {val_auc:.4f}")
    
    clf_full = LogisticRegression(max_iter=1000, C=1.0)
    clf_full.fit(X_train_w2v, y_train)
    joblib.dump(clf_full, 'artifacts/word2vec_lr_model.pkl')
    
    y_test_pred = clf_full.predict_proba(X_test_w2v)[:, 1]
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': y_test_pred
    })
    submission.to_csv('submission_word2vec.csv', index=False)
    print(f"Word2Vec提交文件已保存: submission_word2vec.csv")
    
    return val_auc

def blend_submissions():
    print("\n" + "="*50)
    print("融合多个提交文件...")
    
    from scipy.stats import rankdata
    
    subs = [
        pd.read_csv('submission_tfidf_nbsvm.csv'),
        pd.read_csv('submission_word2vec.csv')
    ]
    
    test_df = pd.read_csv('test_cleaned.csv')
    
    p0 = subs[0]['sentiment'].values
    p1 = subs[1]['sentiment'].values
    
    r0 = rankdata(p0) / len(p0)
    r1 = rankdata(p1) / len(p1)
    
    p_blend = (r0 + r1) / 2
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': p_blend
    })
    submission.to_csv('submission_blend.csv', index=False)
    print(f"融合提交文件已保存: submission_blend.csv")
    print(f"融合方法: rank_mean")

if __name__ == '__main__':
    auc1 = train_tfidf_nbsvm()
    auc2 = train_word2vec_features()
    blend_submissions()
    
    print("\n" + "="*50)
    print("实验总结:")
    print(f"TF-IDF + NBSVM AUC: {auc1:.4f}")
    print(f"Word2Vec + LR AUC: {auc2:.4f}")
    print("融合模型已生成: submission_blend.csv")
