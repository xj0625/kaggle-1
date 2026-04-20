import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("读取数据...")
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv', sep='\t')
unlabeled_df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', on_bad_lines='warn')

print(f"训练数据: {len(train_df)}")
print(f"测试数据: {len(test_df)}")
print(f"无标签数据: {len(unlabeled_df)}")

print("\n清洗训练数据...")
train_df['cleaned_review'] = train_df['review'].apply(clean_text)

print("清洗测试数据...")
test_df['cleaned_review'] = test_df['review'].apply(clean_text)

print("清洗无标签数据...")
unlabeled_df['cleaned_review'] = unlabeled_df['review'].apply(clean_text)

train_df.to_csv('train_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)
unlabeled_df[['review', 'cleaned_review']].to_csv('unlabeled_cleaned.csv', index=False)

print("\n文本预处理完成！")
print(f"训练样本示例:\n{train_df['cleaned_review'].iloc[0][:200]}")
