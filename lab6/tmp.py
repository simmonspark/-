import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.decomposition import TruncatedSVD, PCA
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import random
import re

# 데이터 준비
categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
texts = newsgroups.data

# 텍스트 전처리
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

cleaned_texts = [preprocess_text(text) for text in texts]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(cleaned_texts)

# LDA
tokenized_texts = [text.split() for text in cleaned_texts]
dictionary = Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

lda_model = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=10)
doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
lda_distributions = np.zeros((len(doc_topics), lda_model.num_topics))
for i, doc in enumerate(doc_topics):
    for topic_id, prob in doc:
        lda_distributions[i, topic_id] = prob

# LSA
svd_model = TruncatedSVD(n_components=3, random_state=42)
X_svd = svd_model.fit_transform(X_tfidf)

# BERTopic
bertopic_model = BERTopic()
topics, probs = bertopic_model.fit_transform(cleaned_texts)

# 확률 분포로 정규화 함수
def normalize_to_distribution(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums

# 각 모델의 결과를 확률 분포 형태로 변환
lsa_distributions = np.abs(X_svd)
lsa_distributions = normalize_to_distribution(lsa_distributions)

# BERTopic 확률 분포 확인 및 처리
if probs is None or len(probs.shape) < 2 or probs.shape[1] < 2:
    print("BERTopic probabilities are invalid. Assigning one-hot encoding as fallback.")
    dominant_topics = np.array(topics)
    num_topics = max(dominant_topics) + 1  # 발견된 토픽 수에 맞게 동적으로 설정
    print(f"Detected {num_topics} topics in BERTopic.")
    probs = np.eye(num_topics)[dominant_topics]
else:
    # probs가 유효하면 확률 분포로 정규화
    probs = normalize_to_distribution(probs)

# PCA 및 시각화 함수
def plot_pca_scatter(model_name, topic_distributions, num_topics=3):
    """
    PCA를 통해 차원 축소 후 문서를 2D scatter plot으로 시각화.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(topic_distributions)
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(num_topics):
        plt.scatter(
            reduced_data[np.argmax(topic_distributions, axis=1) == i, 0],
            reduced_data[np.argmax(topic_distributions, axis=1) == i, 1],
            label=f"Topic {i}",
            alpha=0.6,
            c=colors[i]
        )
    plt.title(f"{model_name} - PCA Scatter Plot")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()

# 각 모델의 결과를 시각화
plot_pca_scatter("LDA", lda_distributions)
plot_pca_scatter("LSA", lsa_distributions)
plot_pca_scatter("BERTopic", probs)


# 각 모델의 결과를 시각화
plot_pca_scatter("LDA", lda_distributions)
plot_pca_scatter("LSA", lsa_distributions)
plot_pca_scatter("BERTopic", probs)