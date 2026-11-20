# Portfolio Project Data — Jakarta Hotel Review Sentiment Analysis

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/jakarta_hotel_review_sentiment/`
- **Required Local Images:** `banner.webp`, `image01.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "jakarta-hotel-review-sentiment-analysis",
  "code": "PRO-01 // ML-01",
  "title": "Jakarta Hotel Review Sentiment Analysis",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/jakarta-hotel-review-sentiment-analysis",
  "license": "MIT License",
  "banner": "/images/projects/jakarta_hotel_review_sentiment/banner.webp",
  "summary": "Supervised NLP classification pipeline using Linear SVC and TF-IDF to automate sentiment detection from Jakarta hotel reviews via Streamlit.",
  "projectDescription": "An end-to-end NLP sentiment classification system analyzing guest feedback across South Jakarta hotels. Utilizing SerpAPI web scraping, NLTK text normalization, TF-IDF vectorization, and a Linear Support Vector Classifier, the platform delivers 82.5% test accuracy within an interactive web UI.",
  "architectureSummary": "SerpAPI Review Scraping -> NLTK Lemmatization -> TF-IDF Vectorizer -> Linear SVC -> Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "NLTK",
    "Pandas",
    "NumPy",
    "Streamlit",
    "SerpAPI"
  ],
  "telemetryMatrix": {
    "projectType": "Sentiment Classification System",
    "targetPlatform": "Streamlit Web Application",
    "primaryDomain": "Hospitality NLP Analytics",
    "architecturePipeline": "TF-IDF + Linear SVC Pipeline",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Sentiment Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Hospitality Review NLP",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Text Sentiment Classification",
      "Automated Web Scraping",
      "Interactive Model Serving"
    ]
  },
  "mlProblemContext": "Hospitality businesses struggle to manually aggregate and analyze high-volume customer feedback across digital booking platforms. Inconsistent reviewer vocabulary, subtle negative nuance, and subjective ratings obscure actionable operational insights, necessitating an automated text classification pipeline.",
  "mlDatasetOverview": {
    "profileBadge": "SerpAPI Hotel Reviews",
    "source": "Google Maps Reviews extracted via SerpAPI",
    "modality": "NLP / Text",
    "scale": "210 Customer Reviews across 7 Hotels",
    "inputShape": "210 samples x 672 TF-IDF features",
    "dataFormat": "CSV",
    "description": "Customer reviews collected from 7 premier hotels in South Jakarta with 30 reviews each, categorized into Positive (ratings 4-5) and Negative (ratings 1-3) sentiments.",
    "provenanceNote": "Publicly available Google Maps reviews collected in 2023 via SerpAPI query engine."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Text Feature Space",
    "spaceNote": "Sparse 672-dimensional TF-IDF unigram representation",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "label",
      "type": "Binary Class",
      "distribution": "Positive: 147 (70%), Negative: 63 (30%)"
    },
    "inputSignals": [
      {
        "name": "cleaned_text",
        "type": "Text / NLP Tokens",
        "importanceOrRole": "Primary Input Signal",
        "description": "Lemmatized unigrams extracted from customer reviews after removing punctuation and English stopwords except 'not'."
      }
    ],
    "dataCharacteristicsSummary": "Dataset demonstrates class imbalance with 70% positive and 30% negative sentiment reflecting typical customer satisfaction skew. TF-IDF unigram weighting captures decisive affective keywords such as 'outstanding', 'comfortable', 'disappointed', and 'bad'."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Text Classification",
    "methodologyDescription": "Formulated as a supervised binary classification task mapping text reviews to sentiment polarity. Support Vector Classification with a linear kernel was selected for its proven mathematical optimality, margin maximization, and robust handling of high-dimensional sparse TF-IDF text vectors.",
    "justification": "Linear SVC achieves superior separation boundaries in sparse text spaces without overfitting, outperforming deep neural approaches on small sample regimes while maintaining instantaneous inference latency."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Text Sanitization & Case Normalization",
      "description": "Applied regular expressions to strip non-alphabetical characters and unified tokens into lowercase representation.",
      "badge": "Regex & Lowercase"
    },
    {
      "step": "02",
      "title": "Stopword Filtering with Negation Retention",
      "description": "Filtered standard NLTK English stopwords while explicitly retaining 'not' to protect negative context inversion.",
      "badge": "Stopwords Filtering"
    },
    {
      "step": "03",
      "title": "WordNet Morphological Lemmatization",
      "description": "Transformed inflected token variations into normalized dictionary lemmas using NLTK WordNetLemmatizer.",
      "badge": "Lemmatization"
    },
    {
      "step": "04",
      "title": "TF-IDF Sparse Matrix Construction",
      "description": "Projected cleaned texts into 672-dimensional feature space using TfidfVectorizer with max_df=0.5 and min_df=2.",
      "badge": "TF-IDF Vectorization"
    }
  ],
  "mlDataPreparationNote": "Train/test split of 70/30 (147 train, 63 test) was executed prior to TF-IDF vectorizer fitting to strictly prevent data leakage across evaluation boundaries.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Linear Support Vector Classifier (Linear SVC)",
    "architectureSummary": "Maximum-margin hyperplane classifier operating on L2-normalized 672-dimensional TF-IDF feature space with linear kernel formulation.",
    "configurationBadge": "SVC Linear Kernel",
    "hyperparameters": [
      {
        "name": "kernel",
        "value": "linear",
        "description": "Linear dot-product kernel separating sparse text feature representations"
      },
      {
        "name": "C",
        "value": "1.0",
        "description": "Regularization parameter controlling margin maximization versus slack penalty"
      },
      {
        "name": "random_state",
        "value": "10",
        "description": "Random seed ensuring deterministic coordinate descent optimization"
      },
      {
        "name": "max_df",
        "value": "0.5",
        "description": "Upper document frequency threshold filtering ubiquitous unigrams"
      },
      {
        "name": "min_df",
        "value": "2",
        "description": "Lower document frequency threshold excluding idiosyncratic hapax legomena"
      }
    ],
    "objectiveCriterion": "Hinge Loss with L2 Regularization",
    "optimizationSolver": "Coordinate Descent (liblinear)",
    "trainingProcess": "Model optimization was conducted across 147 training instances using coordinate descent on the primal formulation. The linear decision boundary achieved empirical convergence with zero training error, establishing maximal separation margins between positive and negative vocabulary distributions."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Holdout Test Validation (70/30)",
    "description": "Empirical benchmarking performed on an independent 30% test holdout (63 reviews: 42 positive, 21 negative) with random state seed 0.",
    "metrics": [
      {
        "label": "Accuracy",
        "value": "82.54%",
        "benchmark": "> 80.0%",
        "pass": true
      },
      {
        "label": "Precision",
        "value": "83.67%",
        "benchmark": "> 75.0%",
        "pass": true
      },
      {
        "label": "Recall",
        "value": "76.19%",
        "benchmark": "> 70.0%",
        "pass": true
      },
      {
        "label": "F1-Score",
        "value": "78.24%",
        "benchmark": "> 75.0%",
        "pass": true
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Data Extraction & Inference Workflow",
      "badge": "Application & Data Preview",
      "imageUrl": "/images/projects/jakarta_hotel_review_sentiment/image01.webp",
      "caption": "End-to-end interface showcasing SerpAPI hotel review dataset extraction in Colab and real-time Streamlit sentiment prediction inference.",
      "telemetryNote": "Test Accuracy: 82.54% // Macro F1: 78.24%",
      "tableData": [
        {
          "metric": "Accuracy",
          "trainingScore": "100.0%",
          "validationScore": "82.54%"
        },
        {
          "metric": "Macro Precision",
          "trainingScore": "100.0%",
          "validationScore": "83.67%"
        },
        {
          "metric": "Macro Recall",
          "trainingScore": "100.0%",
          "validationScore": "76.19%"
        },
        {
          "metric": "Macro F1-Score",
          "trainingScore": "100.0%",
          "validationScore": "78.24%"
        }
      ]
    }
  ],
  "mlVisualizationTelemetryNote": "High training accuracy paired with 82.54% holdout generalization confirms effective hyperplane separation with bounded variance.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Decision hyperplane coefficients reveal that positive sentiment is driven by hospitality superlatives ('clean', 'comfortable', 'great', 'outstanding'), while negative classifications are triggered by service delays, hygiene complaints, and negation constructs ('not', 'bad', 'poor').",
    "keyDrivers": [
      {
        "factor": "Hospitality & Comfort Lexicon",
        "impact": "High positive weights on 'clean', 'comfortable', 'great', and 'helpful'"
      },
      {
        "factor": "Service Dissatisfaction & Negation Tokens",
        "impact": "Decisive negative weights on 'not', 'disappointed', 'dirty', and 'bad'"
      }
    ],
    "limitations": [
      {
        "title": "Monolingual English Constraint",
        "description": "The current NLP pipeline exclusively supports English text and cannot classify Bahasa Indonesia reviews or informal slang."
      },
      {
        "title": "Moderate Dataset Scale",
        "description": "Trained on 210 reviews across 7 hotels, limiting generalization to niche hospitality segments or non-urban resort feedback."
      }
    ],
    "futureImprovements": [
      {
        "title": "Multilingual NLP Integration",
        "description": "Incorporate automated translation APIs or fine-tune IndoBERT for seamless Indonesian hospitality sentiment processing.",
        "target": "IndoBERT / Multilingual Pipeline"
      },
      {
        "title": "Dynamic Continuous Scraping",
        "description": "Expand SerpAPI scraper into automated pipelines collecting 5,000+ reviews across nationwide hotel chains.",
        "target": "Expanded 5K+ Review Dataset"
      }
    ],
    "discoveryNote": "Linear hyperplane weights offer complete transparency into word-level sentiment attribution."
  },
  "mlModelArtifacts": [
    {
      "name": "sentiment_analysis_model.pkl",
      "format": "PKL",
      "size": "69 KB",
      "description": "Serialized Python tuple containing fitted TfidfVectorizer and trained Linear SVC classifier ready for inference."
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "Linear Support Vector Classifier",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Holdout Test Validation"
  }
}
```
