import re
from stopwords import get_stopwords_without_negation

# 文本预处理函数
def preprocess_text(text):
    # 1. 去除HTML标签，特别是<br /><br />
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 小写化
    text = text.lower()
    
    # 3. 标点处理：保留情感相关的标点，处理缩写
    # 保留感叹号和问号，处理撇号
    text = re.sub(r"'s", ' is', text)
    text = re.sub(r"'t", ' not', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'m", ' am', text)
    text = re.sub(r"'ll", ' will', text)
    
    # 4. 移除其他标点符号，只保留感叹号和问号
    text = re.sub(r'[^a-zA-Z0-9\s!\?]', ' ', text)
    
    # 5. 分词
    words = text.split()
    
    # 6. 停用词过滤（保留否定词）
    stopwords = get_stopwords_without_negation()
    words = [word for word in words if word not in stopwords]
    
    # 7. 重新组合文本
    processed_text = ' '.join(words)
    
    return processed_text

# 测试函数
def test_preprocess():
    test_text = "This is a <br /> test! Don't you think it's great?"
    result = preprocess_text(test_text)
    print(f"原始文本: {test_text}")
    print(f"处理后: {result}")

if __name__ == "__main__":
    test_preprocess()
