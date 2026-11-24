import os
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

figures_dir = os.path.join(os.getcwd(), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Preprocessing function matching the original logic
def clean_and_lemmatize(text):
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = cleaned_text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in set(all_stopwords)]
    return " ".join(lemmatized_tokens)

# Load data
df = pd.read_csv("reviewHotelJakarta.csv", encoding="latin-1")
df['cleaned_text'] = df['review'].apply(clean_and_lemmatize)
df['label'] = df['rating'].map({
    1.0: 'Negative Review',
    2.0: 'Negative Review',
    3.0: 'Negative Review',
    4.0: 'Positive Review',
    5.0: 'Positive Review'
})
df['char_length'] = df['review'].apply(lambda x: len(str(x)))
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

# -------------------------------------------------------------
# 1. Rating Distribution
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
rating_counts = df['rating'].value_counts().sort_index()
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color=palette, edgecolor='#333333', linewidth=0.8, width=0.55)

for bar in bars:
    height = bar.get_height()
    pct = (height / len(df)) * 100
    ax.annotate(f'{height}\n({pct:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

ax.set_title('Distribusi Rating Hotel di Jakarta (Skala 1.0 - 5.0)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Rating Bintang', fontsize=11, fontweight='bold')
ax.set_ylabel('Jumlah Ulasan', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(rating_counts.values) * 1.18)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '01_rating_distribution.png'))
plt.close(fig)
print("Saved 01_rating_distribution.png")

# -------------------------------------------------------------
# 2. Sentiment Distribution (Positive vs Negative)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
sentiment_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

wedges, texts, autotexts = ax.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode,
    pctdistance=0.75,
    textprops={'fontsize': 11, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.5}  # Donut chart
)
for autotext in autotexts:
    autotext.set_color('#ffffff')
    autotext.set_fontsize(12)

ax.set_title(f'Proporsi Sentimen Ulasan Hotel\n(Total: {len(df)} Ulasan)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '02_sentiment_distribution.png'))
plt.close(fig)
print("Saved 02_sentiment_distribution.png")

# -------------------------------------------------------------
# 3. Hotel Ratings & Review Count Comparison
# -------------------------------------------------------------
hotel_stats = df.groupby('Hotel_name').agg(
    avg_rating=('rating', 'mean'),
    review_count=('rating', 'count')
).reset_index().sort_values(by='avg_rating', ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Avg Rating Bar
bars1 = ax1.barh(hotel_stats['Hotel_name'], hotel_stats['avg_rating'], color='#3498db', edgecolor='#2980b9', height=0.55)
for bar in bars1:
    width = bar.get_width()
    ax1.annotate(f'{width:.2f}',
                 xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=9, fontweight='bold')
ax1.set_xlim(0, 5.5)
ax1.set_title('Rata-rata Rating per Hotel', fontsize=12, fontweight='bold')
ax1.set_xlabel('Rata-rata Rating (Skala 1-5)', fontsize=10, fontweight='bold')

# Review Count Bar
bars2 = ax2.barh(hotel_stats['Hotel_name'], hotel_stats['review_count'], color='#9b59b6', edgecolor='#8e44ad', height=0.55)
for bar in bars2:
    width = bar.get_width()
    ax2.annotate(f'{int(width)}',
                 xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, max(hotel_stats['review_count']) * 1.25)
ax2.set_title('Total Jumlah Ulasan per Hotel', fontsize=12, fontweight='bold')
ax2.set_xlabel('Jumlah Ulasan', fontsize=10, fontweight='bold')
ax2.set_yticks([])

plt.suptitle('Analisis Perbandingan Hotel di Jakarta', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '03_hotel_ratings_and_reviews.png'))
plt.close(fig)
print("Saved 03_hotel_ratings_and_reviews.png")

# -------------------------------------------------------------
# 4. Review Length Distribution by Sentiment
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

sns.boxplot(x='label', y='char_length', data=df, ax=ax1, hue='label', legend=False,
            palette={'Positive Review': '#2ecc71', 'Negative Review': '#e74c3c'})
ax1.set_title('Distribusi Panjang Karakter per Sentimen', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentimen', fontsize=10, fontweight='bold')
ax1.set_ylabel('Jumlah Karakter', fontsize=10, fontweight='bold')

sns.boxplot(x='label', y='word_count', data=df, ax=ax2, hue='label', legend=False,
            palette={'Positive Review': '#2ecc71', 'Negative Review': '#e74c3c'})
ax2.set_title('Distribusi Jumlah Kata per Sentimen', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sentimen', fontsize=10, fontweight='bold')
ax2.set_ylabel('Jumlah Kata', fontsize=10, fontweight='bold')

plt.suptitle('Analisis Panjang Teks Ulasan Berdasarkan Sentimen', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '04_review_length_distribution.png'))
plt.close(fig)
print("Saved 04_review_length_distribution.png")

# -------------------------------------------------------------
# 5 & 6. Top Frequent Words in Positive vs Negative Reviews
# -------------------------------------------------------------
def get_top_words(texts, n=15):
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.split())
    series = pd.Series(all_tokens).value_counts()
    return series.head(n)

pos_top = get_top_words(df[df['label'] == 'Positive Review']['cleaned_text'], 15)
neg_top = get_top_words(df[df['label'] == 'Negative Review']['cleaned_text'], 15)

# Positive top words
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
bars = ax.barh(pos_top.index[::-1], pos_top.values[::-1], color='#27ae60', edgecolor='#1e824c', height=0.6)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{int(w)}', xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 15 Kata Paling Sering Muncul pada Ulasan Positif', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Frekuensi Kemunculan', fontsize=10, fontweight='bold')
ax.set_xlim(0, max(pos_top.values) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '05_top_words_positive.png'))
plt.close(fig)
print("Saved 05_top_words_positive.png")

# Negative top words
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
bars = ax.barh(neg_top.index[::-1], neg_top.values[::-1], color='#c0392b', edgecolor='#962d22', height=0.6)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{int(w)}', xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 15 Kata Paling Sering Muncul pada Ulasan Negatif', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Frekuensi Kemunculan', fontsize=10, fontweight='bold')
ax.set_xlim(0, max(neg_top.values) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '06_top_words_negative.png'))
plt.close(fig)
print("Saved 06_top_words_negative.png")

# -------------------------------------------------------------
# Train / Fit Exact Model as in sentiment_analysis_model.ipynb
# -------------------------------------------------------------
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
tfidf_train = tfidf.fit_transform(X_train)
classifier = SVC(kernel='linear', random_state=10)
classifier.fit(tfidf_train, y_train)

# Re-save model pickle so it is fresh and fully functional across versions
with open("sentiment_analysis_model.pkl", "wb") as model_file:
    pickle.dump((tfidf, classifier), model_file)
print("Updated sentiment_analysis_model.pkl")

# -------------------------------------------------------------
# 7. Model Evaluation & Confusion Matrix
# -------------------------------------------------------------
X_test_vec = tfidf.transform(X_test).toarray()
y_pred = classifier.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred, labels=['Positive Review', 'Negative Review'])
acc = accuracy_score(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Positive', 'Pred: Negative'],
            yticklabels=['Actual: Positive', 'Actual: Negative'],
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
ax.set_title(f'Confusion Matrix Model SVM (Linear)\nAkurasi Uji: {acc*100:.2f}%', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '07_model_confusion_matrix.png'))
plt.close(fig)
print("Saved 07_model_confusion_matrix.png")

# -------------------------------------------------------------
# 8. SVM Feature Importance (Top Indicative Words)
# -------------------------------------------------------------
feature_names = np.array(tfidf.get_feature_names_out())
coef = classifier.coef_.toarray()[0] if hasattr(classifier.coef_, 'toarray') else classifier.coef_[0]

# Positive class alignment
pos_class_index = 1 if classifier.classes_[1] == 'Positive Review' else 0
if pos_class_index == 0:
    coef = -coef

top_pos_idx = np.argsort(coef)[-10:]
top_neg_idx = np.argsort(coef)[:10]

top_features = np.concatenate([top_neg_idx, top_pos_idx])
top_feature_names = feature_names[top_features]
top_coefficients = coef[top_features]

colors_feat = ['#e74c3c' if c < 0 else '#2ecc71' for c in top_coefficients]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars = ax.barh(top_feature_names, top_coefficients, color=colors_feat, edgecolor='#333333', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Top Kata Paling Berpengaruh dalam Klasifikasi Sentimen (SVM Coef)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Koefisien SVM ( <0: Indikator Negatif, >0: Indikator Positif )', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, '08_svm_feature_importance.png'))
plt.close(fig)
print("Saved 08_svm_feature_importance.png")

print("All visualizations generated successfully in 'figures' directory!")
