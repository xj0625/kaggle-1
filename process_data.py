import pandas as pd
from preprocess import preprocess_text

# 读取训练数据
df = pd.read_csv('labeledTrainData.tsv', sep='\t')

# 对评论进行预处理
df['processed_review'] = df['review'].apply(preprocess_text)

# 保存处理后的数据
df.to_csv('processed_train_data.csv', index=False)

print(f"处理完成，共处理了 {len(df)} 条评论")
print("处理后的数据已保存到 processed_train_data.csv")

# 显示前5条处理结果
print("\n前5条处理结果：")
print(df[['sentiment', 'processed_review']].head())
