stopwords = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
    'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'may', 'might', 'must', 'can', 'cannot', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'who',
    'whom', 'whose', 'which', 'what', 'when', 'where', 'why', 'how', 'all',
    'each', 'every', 'some', 'any', 'few', 'more', 'most', 'other', 'another',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
    'eighth', 'ninth', 'tenth', 'last', 'next', 'then', 'now', 'before',
    'after', 'during', 'while', 'because', 'since', 'until', 'if', 'unless',
    'although', 'though', 'even', 'also', 'too', 'only', 'just', 'already',
    'still', 'yet', 'again', 'further', 'so', 'very', 'really', 'quite',
    'rather', 'such', 'both', 'either', 'neither', 'not', 'no', 'never', 'nor'
])

# 注意：否定词如not, no, never, nor已保留在列表中，在实际使用时需要根据情况决定是否移除

# 移除否定词的函数
def get_stopwords_without_negation():
    negation_words = {'not', 'no', 'never', 'nor'}
    return stopwords - negation_words

# 完整的停用词列表（包含否定词）
def get_full_stopwords():
    return stopwords
