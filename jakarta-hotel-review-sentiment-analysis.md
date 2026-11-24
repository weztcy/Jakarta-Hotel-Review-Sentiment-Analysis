# Portfolio Project Data — Jakarta Hotel Review Sentiment Analysis

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/jakarta_hotel_review_sentiment/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image08.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "jakarta-hotel-review-sentiment-analysis",
  "code": "PRO-14 // ML-01",
  "title": "Jakarta Hotel Review Sentiment Analysis",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Jakarta-Hotel-Review-Sentiment-Analysis",
  "license": "MIT License",
  "banner": "/images/projects/jakarta_hotel_review_sentiment/banner.webp",
  "summary": "End-to-end NLP sentiment classifier using TF-IDF and Linear SVC on scraped Jakarta hotel reviews, served via an interactive Streamlit web app.",
  "projectDescription": "Built an end-to-end NLP sentiment analysis pipeline that scrapes 210 Google Maps reviews across seven Jakarta hotels via SerpAPI, preprocesses text with NLTK lemmatization and TF-IDF vectorization, and serves Linear SVC binary predictions (82.54% test accuracy) in Streamlit.",
  "architectureSummary": "SerpAPI Scraping -> NLTK Regex & Lemmatization -> TF-IDF Vectorization -> Linear SVC Classifier -> Pickle Bundle -> Streamlit Inference UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "NLTK",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "SerpAPI",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised NLP Classification",
    "targetPlatform": "Python & Streamlit Web App",
    "primaryDomain": "Hospitality Sentiment Analytics",
    "architecturePipeline": "TF-IDF + Linear SVC Pipeline",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "NLP Sentiment Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Hospitality Review Text Analytics",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "TF-IDF Vectorization",
      "Linear SVC Classifier",
      "Streamlit Live Inference"
    ]
  },
  "mlProblemContext": "Manual evaluation of hospitality feedback across Jakarta hotels is slow, inconsistent, and vulnerable to rating bias or varied phrasing. Automating polarity detection helps hotel managers and travelers extract reliable service, cleanliness, and facility insights from unstructured review text.",
  "mlDatasetOverview": {
    "profileBadge": "Google Maps Reviews Corpus",
    "source": "SerpAPI Google Maps Reviews (7 Jakarta Hotels)",
    "modality": "NLP",
    "scale": "210 Reviews (30 per Hotel; 147 Train / 63 Test)",
    "inputShape": "N x 672 TF-IDF Unigram Features",
    "dataFormat": "CSV",
    "description": "Balanced 30-review samples per property across Hotel Gran Mahakam, Veranda Hotel at Pakubuwono, Sutasoma Hotel, The Dharmawangsa Jakarta, Hotel Monopoli, grandkemang Jakarta, and Park Regis Arion, mapped into 147 positive and 63 negative reviews.",
    "provenanceNote": "Collected programmatically via SerpAPI Google Maps engine and archived in reviewHotelJakarta.csv with zero null entries."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Lexical TF-IDF Feature Space",
    "spaceNote": "672 unigram terms retained after max_df=0.5 and min_df=2 pruning on the 147-review training split.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Sentiment Polarity Label",
      "type": "Binary Class",
      "distribution": "70.0% Positive Review (147) vs. 30.0% Negative Review (63)"
    },
    "inputSignals": [
      {
        "name": "Lemmatized Review TF-IDF Matrix",
        "type": "Sparse Numerical Matrix",
        "importanceOrRole": "Primary predictive text representation (672 features)",
        "description": "TF-IDF weighted unigrams generated from cleaned, WordNet-lemmatized English review text with negation token 'not' preserved."
      },
      {
        "name": "Star Rating Ground Truth",
        "type": "Ordinal Supervision Signal",
        "importanceOrRole": "Binary label mapping source (1.0-5.0 scale)",
        "description": "Original Google Maps star ratings binarized into Negative Review (1.0-3.0) and Positive Review (4.0-5.0)."
      },
      {
        "name": "Negation Token ('not')",
        "type": "Preserved Lexical Indicator",
        "importanceOrRole": "Strongest negative coefficient (-1.7070)",
        "description": "Explicitly excluded from NLTK English stopword removal so negative phrases remain separable in linear SVM weights."
      },
      {
        "name": "Review Length & Verbosity",
        "type": "Exploratory Text Characteristic",
        "importanceOrRole": "Diagnostic distribution signal",
        "description": "Negative reviews average 103.4 words (median 86) versus 51.9 words (median 36) for positive reviews."
      }
    ],
    "dataCharacteristicsSummary": "The corpus exhibits a 70% to 30% positive-to-negative class split with 47.1% five-star ratings. Negative reviews are notably more verbose, averaging 103.4 words and 572.5 characters compared to 51.9 words and 295.3 characters for positive reviews, with zero missing values across all records."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Binary Text Classification",
    "methodologyDescription": "Reviews are mapped to binary sentiment classes, normalized via regex filtering, custom negation-preserving stopword removal, and WordNet lemmatization, then transformed into a 672-term TF-IDF matrix before fitting a linear-kernel Support Vector Classifier on a 70/30 train-test split.",
    "justification": "On a compact 210-document hospitality corpus, TF-IDF paired with a maximum-margin Linear SVC avoids overfitting compared to high-parameter neural networks while yielding directly interpretable feature coefficients."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Binary Sentiment Mapping",
      "description": "Converted 5-point Google Maps star ratings into binary classes, assigning ratings 4.0 and 5.0 to Positive Review and ratings 1.0 through 3.0 to Negative Review.",
      "badge": "Label Encoding"
    },
    {
      "step": "02",
      "title": "Regex & Case Normalization",
      "description": "Removed non-alphabetic symbols and punctuation using [^a-zA-Z] substitution and lowercased all review strings prior to tokenization.",
      "badge": "Text Cleaning"
    },
    {
      "step": "03",
      "title": "Negation-Preserving Stopword Removal & Lemmatization",
      "description": "Filtered NLTK English stopwords while retaining 'not', then normalized remaining tokens to dictionary base forms via WordNetLemmatizer.",
      "badge": "NLTK Normalization"
    },
    {
      "step": "04",
      "title": "Leakage-Free TF-IDF Vectorization",
      "description": "Fitted TfidfVectorizer(max_df=0.5, min_df=2) solely on the 147-sample training split (random_state=0) to produce 672 vocabulary features.",
      "badge": "Feature Extraction"
    }
  ],
  "mlDataPreparationNote": "TfidfVectorizer is fitted exclusively on X_train after the 70/30 train_test_split (random_state=0) so test-set token frequencies never leak into training weights.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Linear Support Vector Classifier (SVC)",
    "architectureSummary": "Maximum-margin linear hyperplane classifier operating over 672 sparse TF-IDF unigram features, serialized jointly with the fitted vectorizer as a (tfidf, classifier) pickle tuple.",
    "configurationBadge": "Linear Kernel SVM",
    "hyperparameters": [
      {
        "name": "kernel",
        "value": "linear",
        "description": "Separates positive and negative TF-IDF vectors with an interpretable linear hyperplane."
      },
      {
        "name": "max_df",
        "value": "0.5",
        "description": "Excludes overly frequent tokens appearing in more than 50% of training reviews."
      },
      {
        "name": "min_df",
        "value": "2",
        "description": "Filters singleton noise terms occurring in fewer than 2 training documents."
      },
      {
        "name": "test_size",
        "value": "0.3",
        "description": "Allocates 147 reviews for training and 63 reviews for held-out test evaluation."
      },
      {
        "name": "random_state",
        "value": "10 (SVC) / 0 (Split)",
        "description": "Fixes deterministic train-test partitioning and SVM solver initialization."
      }
    ],
    "objectiveCriterion": "Hinge Loss with L2 Margin Regularization",
    "optimizationSolver": "LIBLINEAR / Sequential Minimal Optimization (SMO)",
    "trainingProcess": "The 147 training reviews are vectorized with TfidfVectorizer (max_df=0.5, min_df=2) fitted solely on training text to prevent data leakage. A linear SVC (random_state=10) then optimizes the maximum-margin hyperplane over 672 unigram features and is serialized alongside the vectorizer as a pickle tuple."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "70/30 Hold-Out Test Split Evaluation",
    "description": "Model generalization is evaluated on the unseen 63-review test partition (42 positive, 21 negative) using accuracy, per-class precision, recall, F1-score, confusion matrix counts, and decision-function ROC-AUC.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "82.54%",
        "benchmark": "Baseline: 70.00%"
      },
      {
        "label": "ROC-AUC SCORE",
        "value": "0.9150",
        "benchmark": "Target > 0.85"
      },
      {
        "label": "POSITIVE F1",
        "value": "0.8791",
        "benchmark": "Threshold >= 0.80"
      },
      {
        "label": "NEGATIVE F1",
        "value": "0.6857",
        "benchmark": "Target > 0.65"
      },
      {
        "label": "WEIGHTED F1",
        "value": "0.8147",
        "benchmark": "Threshold >= 0.80"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Live Sentiment Prediction Interface",
      "badge": "INFERENCE UI",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image01.webp",
      "telemetryNote": "Interactive Streamlit web application executing real-time preprocessing and Linear SVC classification on guest review text."
    },
    {
      "title": "Jakarta Hotel Star Rating Distribution (1.0 - 5.0)",
      "badge": "EDA BAR CHART",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image02.webp",
      "telemetryNote": "47.1% 5.0-star (99) and 22.9% 4.0-star (48) reviews dominate the 210-sample corpus over 1.0-3.0 ratings."
    },
    {
      "title": "Linear SVM Confusion Matrix & Class Proportion",
      "badge": "CONFUSION MATRIX",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image03.webp",
      "telemetryNote": "82.54% test accuracy (40 TP, 12 TN, 9 FP, 2 FN) alongside the 70.0% positive vs. 30.0% negative corpus split."
    },
    {
      "title": "Hotel Average Rating & Sample Count Comparison",
      "badge": "PROPERTY BENCHMARK",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image04.webp",
      "telemetryNote": "Equal 30-review sampling across 7 properties, spanning Hotel Gran Mahakam (4.80 avg) to grandkemang Jakarta (2.90 avg)."
    },
    {
      "title": "Review Character Length & Word Count by Sentiment",
      "badge": "LENGTH BOXPLOTS",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image05.webp",
      "telemetryNote": "Negative reviews show higher verbosity (median 468 chars / 86 words) than positive reviews (median 212 chars / 36 words)."
    },
    {
      "title": "Linear SVM Top Indicative Feature Coefficients",
      "badge": "FEATURE IMPORTANCE",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image06.webp",
      "telemetryNote": "Weights highlight 'clean' (+0.85) and 'enough' (+0.84) for positive polarity vs. 'not' (-1.71) and 'sleep' (-1.53) for negative."
    },
    {
      "title": "Top 15 Frequent Tokens in Positive Reviews",
      "badge": "LEXICAL FREQUENCY",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image07.webp",
      "telemetryNote": "Lemmatized positive corpus led by 'hotel' (149), 'room' (130), 'service' (91), 'location' (67), and 'good' (62)."
    },
    {
      "title": "Top 15 Frequent Tokens in Negative Reviews",
      "badge": "LEXICAL FREQUENCY",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image08.webp",
      "telemetryNote": "Lemmatized negative corpus dominated by 'room' (126), preserved negation 'not' (84), 'hotel' (84), and 'service' (49)."
    }
  ],
  "mlVisualizationTelemetryNote": "Training accuracy reaches 100.00% on the 147-sample training set versus 82.54% on the 63-sample held-out test split, indicating strong positive recall (95.24%) with moderate overfitting on minority negative reviews (57.14% recall).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Linear SVC feature weights directly expose polarity drivers across the 672-word vocabulary. Positive predictions rely on terms like clean, enough, big, restaurant, and good, whereas negative predictions are strongly anchored by negation and complaint tokens such as not, sleep, even, said, and dirty.",
    "keyDrivers": [
      {
        "factor": "Negation Token ('not')",
        "impact": "Strongest negative driver (coef -1.7070), preserving dissatisfaction statements during stopword filtering"
      },
      {
        "factor": "Sleep & Noise Complaints ('sleep', 'noisy', 'night')",
        "impact": "High-magnitude negative weights (-1.5315 for 'sleep', -1.0033 for 'noisy') signaling acoustic discomfort"
      },
      {
        "factor": "Cleanliness & Room Quality ('clean', 'dirty')",
        "impact": "Symmetric polarity anchor (+0.8482 for 'clean' vs. -1.0702 for 'dirty') separating hygiene outcomes"
      },
      {
        "factor": "Hospitality Amenities ('big', 'restaurant', 'pool', 'food')",
        "impact": "Consistent positive coefficients (+0.65 to +0.80) reinforcing favorable guest stay experiences"
      }
    ],
    "limitations": [
      {
        "title": "Minority Negative Class Recall Gap",
        "description": "With a 70/30 class imbalance (147 positive vs. 63 negative), test recall on Negative Review drops to 57.14% with 9 false positives."
      },
      {
        "title": "Three-Star Rating Binarization Ambiguity",
        "description": "Mapping neutral 3.0-star reviews (24 records) to Negative Review introduces mixed-praise phrasing inside the negative training class."
      },
      {
        "title": "English-Only Corpus & Stopword Scope",
        "description": "NLTK English stopwords and WordNet lemmatization restrict accurate sentiment classification strictly to English hotel reviews."
      }
    ],
    "futureImprovements": [
      {
        "title": "Class-Weighted SVM & Bigram TF-IDF",
        "target": "Improve Negative Review recall above 75% using class_weight='balanced' and (1,2) n-grams"
      },
      {
        "title": "Three-Class Neutral Sentiment Modeling",
        "target": "Separate 3.0-star reviews into a dedicated Neutral class to eliminate binary boundary noise"
      },
      {
        "title": "Automated Translation & IndoBERT Support",
        "target": "Handle Bahasa Indonesia and multilingual Jakarta hotel reviews natively"
      }
    ],
    "discoveryNote": "Preserving 'not' in NLTK stopword filtering proved decisive, making 'not' the highest-magnitude single feature in the trained Linear SVC."
  },
  "mlModelArtifacts": [
    {
      "name": "sentiment_analysis_model.pkl",
      "format": "PKL"
    },
    {
      "name": "streamlit/sentiment_analysis_model.pkl",
      "format": "PKL"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Text Classification",
    "modelArchitecture": "Linear SVC with TF-IDF",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Hold-Out Test Split"
  }
}
```
