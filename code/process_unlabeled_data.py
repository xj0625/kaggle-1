import pandas as pd
from preprocess import preprocess_text

# 读取无标签训练数据，处理格式问题
df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')

# 对评论进行预处理
df['processed_review'] = df['review'].apply(preprocess_text)

# 保存处理后的数据
df.to_csv('processed_unlabeled_data.csv', index=False)

print(f"处理完成，共处理了 {len(df)} 条无标签评论")
print("处理后的数据已保存到 processed_unlabeled_data.csv")

# 显示前5条处理结果
print("\n前5条处理结果：")
print(df[['id', 'processed_review']].head())
