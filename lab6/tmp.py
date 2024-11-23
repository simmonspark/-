import nltk
nltk.download('punkt')
nltk.download('stopwords')
# 필요한 라이브러리 불러오기
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.decomposition import TruncatedSVD
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np

# 1. 데이터 준비
categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
texts = newsgroups.data

# 텍스트 전처리 함수 정의
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# 전처리된 데이터 생성
cleaned_texts = [preprocess_text(text) for text in texts]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(cleaned_texts)

# 2. 모델 적용

# 2.1 LDA
tokenized_texts = [text.split() for text in cleaned_texts]
dictionary = Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

lda_model = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=10)
lda_topics = lda_model.print_topics(num_words=10)

# LDA 문서-토픽 분포
doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
topic_distributions = np.zeros((len(doc_topics), lda_model.num_topics))
for i, doc in enumerate(doc_topics):
    for topic_id, prob in doc:
        topic_distributions[i, topic_id] = prob

# 2.2 LSA
svd_model = TruncatedSVD(n_components=3, random_state=42)
X_svd = svd_model.fit_transform(X_tfidf)
terms = vectorizer.get_feature_names_out()
lsa_topics = [
    [terms[idx] for idx in comp.argsort()[-10:]]
    for comp in svd_model.components_
]

# 2.3 BERTopic
bertopic_model = BERTopic()
topics, probs = bertopic_model.fit_transform(cleaned_texts)
bertopic_topics = bertopic_model.get_topics()

import random


# 샘플 문서 선택 및 시각화 (scatter plot 기반)
def plot_dirichlet_distributions_scatter(model_name, topic_distributions, num_samples=10, num_topics=3):
    # 배열 차원 확인 및 2차원 변환
    topic_distributions = np.atleast_2d(topic_distributions)  # 1차원 배열도 2차원으로 변환

    fig, axes = plt.subplots(1, num_topics, figsize=(15, 5))
    fig.suptitle(f"{model_name} - Dirichlet Distributions (Scatter for Sampled Documents)", fontsize=16)

    for topic_id in range(num_topics):
        # 특정 토픽에 속하는 문서 필터링
        topic_indices = np.argsort(topic_distributions[:, topic_id])[::-1]
        sampled_indices = random.sample(list(topic_indices), min(len(topic_indices), num_samples))

        # Scatter plot으로 각 문서의 토픽 확률 시각화
        for doc_id in sampled_indices:
            axes[topic_id].scatter(
                range(len(topic_distributions[doc_id])),  # x축: 토픽 번호
                topic_distributions[doc_id],  # y축: 해당 문서의 토픽 확률
                alpha=0.6,
                label=f"Doc {doc_id}" if doc_id < 5 else ""  # 너무 많은 레이블 방지
            )

        axes[topic_id].set_title(f"Topic {topic_id}")
        axes[topic_id].set_xlabel("Topic Index")
        axes[topic_id].set_ylabel("Probability")
        axes[topic_id].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# 1. LDA 모델 샘플링 및 시각화
lda_distributions = topic_distributions  # LDA 문서-토픽 분포
plot_dirichlet_distributions_scatter("LDA", lda_distributions)

# 2. LSA 모델 샘플링 및 시각화
lsa_distributions = X_svd / X_svd.sum(axis=1, keepdims=True)  # LSA 확률 분포화
plot_dirichlet_distributions_scatter("LSA", lsa_distributions)

# 3. BERTopic 모델 샘플링 및 시각화
bertopic_distributions = np.atleast_2d(probs)
plot_dirichlet_distributions_scatter("BERTopic", bertopic_distributions)
