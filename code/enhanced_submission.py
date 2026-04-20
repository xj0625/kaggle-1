import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import sparse
from scipy.stats import rankdata
import joblib
import os

def get_texts():
    train_df = pd.read_csv('train_cleaned.csv')
    test_df = pd.read_csv('test_cleaned.csv')
    return train_df, test_df

def extract_all_features(train_texts, test_texts):
    print("提取多种TF-IDF特征...")
    
    features = []
    
    vectorizer_word = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'\w{1,}'
    )
    X_word = vectorizer_word.fit_transform(train_texts)
    X_word_test = vectorizer_word.transform(test_texts)
    features.append(('word', X_word, X_word_test, vectorizer_word))
    print(f"Word n-gram: {X_word.shape}")
    
    vectorizer_char = TfidfVectorizer(
        max_features=50000,
        ngram_range=(2,6),
        sublinear_tf=True,
        analyzer='char',
        strip_accents='unicode'
    )
    X_char = vectorizer_char.fit_transform(train_texts)
    X_char_test = vectorizer_char.transform(test_texts)
    features.append(('char', X_char, X_char_test, vectorizer_char))
    print(f"Char n-gram: {X_char.shape}")
    
    return features

class NBSVM:
    def __init__(self, alpha=1.0, C=4.0):
        self.alpha = alpha
        self.C = C
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
        self.clf = LogisticRegression(max_iter=1000, C=self.C, solver='liblinear')
        self.clf.fit(X_new, y)
        return self
    
    def predict_proba(self, X):
        X_new = X.multiply(self.r)
        return self.clf.predict_proba(X_new)

def train_single_model(X_train, X_test, y_train, model_name='nbsvm'):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    if model_name == 'nbsvm':
        model = NBSVM(alpha=1.0, C=4.0)
        model.fit(X_tr, y_tr)
    elif model_name == 'lr':
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_tr, y_tr)
    elif model_name == 'svc':
        svc = LinearSVC(C=1.0, max_iter=2000)
        model = CalibratedClassifierCV(svc, cv=3)
        model.fit(X_tr, y_tr)
    
    y_val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    model_full = type(model)(**model.get_params()) if hasattr(model, 'get_params') else None
    if model_name == 'nbsvm':
        model_full = NBSVM(alpha=1.0, C=4.0)
    elif model_name == 'lr':
        model_full = LogisticRegression(max_iter=1000, C=1.0)
    elif model_name == 'svc':
        svc_full = LinearSVC(C=1.0, max_iter=2000)
        model_full = CalibratedClassifierCV(svc_full, cv=3)
    
    model_full.fit(X_train, y_train)
    
    return model_full, val_auc

def make_submission():
    train_df, test_df = get_texts()
    train_texts = train_df['cleaned_review'].values
    test_texts = test_df['cleaned_review'].values
    y_train = train_df['sentiment'].values
    
    print(f"训练样本: {len(train_texts)}, 测试样本: {len(test_texts)}")
    
    features = extract_all_features(train_texts, test_texts)
    
    all_preds = []
    
    for feat_name, X_train_feat, X_test_feat, vec in features:
        print(f"\n训练 {feat_name} 特征模型...")
        
        for model_name in ['nbsvm', 'lr']:
            print(f"  训练 {model_name}...")
            model, val_auc = train_single_model(X_train_feat, X_test_feat, y_train, model_name)
            print(f"    验证集AUC: {val_auc:.4f}")
            
            y_test_pred = model.predict_proba(X_test_feat)[:, 1]
            all_preds.append((f'{feat_name}_{model_name}', y_test_pred, val_auc))
    
    print("\n" + "="*50)
    print("融合预测结果...")
    
    sorted_preds = sorted(all_preds, key=lambda x: x[2], reverse=True)
    
    top_preds = [p[1] for p in sorted_preds[:5]]
    
    p_rank = np.zeros(len(test_df))
    for pred in top_preds:
        p_rank += rankdata(pred) / len(pred)
    p_rank /= len(top_preds)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': p_rank
    })
    submission.to_csv('submission_final.csv', index=False)
    print(f"最终提交文件已保存: submission_final.csv")
    
    print("\n模型性能排名:")
    for name, _, auc in sorted_preds:
        print(f"  {name}: {auc:.4f}")
    
    submission_best = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': sorted_preds[0][1]
    })
    submission_best.to_csv('submission_best_single.csv', index=False)
    print(f"\n最佳单模型提交: {sorted_preds[0][0]} ({sorted_preds[0][2]:.4f})")

if __name__ == '__main__':
    make_submission()
