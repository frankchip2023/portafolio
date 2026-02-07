import churnProcess from '../assets/images/churn_process.png';
import churnAnalysis from '../assets/images/churn_analysis_objective.png';
import churnInitialData from '../assets/images/churn_initial_data.png';
import churnTargetAnalysis from '../assets/images/churn_target_analysis.png';
import churnSkewness from '../assets/images/churn_skewness.png';
import churnDistribution from '../assets/images/churn_distribution.png';
import churnCategorical from '../assets/images/churn_categorical_analysis.png';
import churnCorrelation from '../assets/images/churn_correlation_matrix.png';
import churnCleaning from '../assets/images/churn_data_cleaning.png';
import churnModels from '../assets/images/churn_models.png';
import churnEvaluation from '../assets/images/churn_model_evaluation.png';
import churnNumerical from '../assets/images/churn_numerical_distribution.png';
import churnFeatureImportance from '../assets/images/churn_feature_importance.png';
import churnMain from '../assets/images/churn_main.png';
import segmentationDatasetHead from '../assets/images/segmentation_dataset_head.png';
import segmentationMethodology from '../assets/images/segmentation_methodology.png';
import segmentationMethodologySecond from '../assets/images/segmentation_methodology_2.png';
import segmentationFeatureEngineering from '../assets/images/segmentation_feature_engineering.png';
import segmentationModelsCompared from '../assets/images/segmentation_models_compared.png';
import segmentationModelingApproachSecond from '../assets/images/segmentation_modeling_approach_2.png';
import segmentationModelingApproachThird from '../assets/images/segmentation_modeling_approach_3.png';
import segmentationProfiling from '../assets/images/segmentation_profiling.png';
import segmentationProfilingSecond from '../assets/images/segmentation_profiling_2.png';
import segmentationCardImage from '../assets/images/segmentation_card.png';
import ppeOverview from '../assets/images/ppe_overview.png';
import ppeMain from '../assets/images/ppe_main.jpg';
import ragProjectScheme from '../assets/images/rag_project_scheme.png';

export interface Project {
    id: string;
    title: string;
    description: string;
    longDescription?: string;
    tags: string[];
    imageUrl: string;
    kaggleUrl?: string;
    githubUrl?: string;
    demoUrl?: string;
    demoTitle?: string;
    codeTitle?: string;
    overviewImage?: string; // Single image to be embedded in description
    projectSchemeImage?: string; // System / topology diagram (RAG deployment)
    datasetSchema?: { category: string; variables: string }[];
    analysisImage?: string; // Image for the Analysis Objective section
    initialDataImage?: string; // Image for Initial Data Inspection
    targetAnalysisImage?: string; // Image for Target Variable Analysis
    targetDistributionImage?: string; // Image for Churn Distribution (Bar Chart)
    skewnessImage?: string; // Image for Skewness Analysis
    numericalDistributionImage?: string; // Image for Numerical Feature Distributions
    categoricalAnalysisImage?: string; // Image for Categorical Variables Analysis
    correlationImage?: string; // Image for Correlation Matrix
    cleaningImage?: string; // Image for Data Cleaning Process
    modelsImage?: string; // Image for Machine Learning Models
    modelsImageSecondary?: string; // Optional second image for modeling section
    modelsImageTertiary?: string; // Optional third image for modeling section
    profilingImage?: string;
    profilingImageSecondary?: string;
    evaluationImage?: string; // Image for Model Evaluation Metrics
    featureImportanceImage?: string; // Image for Feature Importance
    datasetTitle?: string; // Custom title for the dataset button (e.g. "Roboflow")
}

export const projects: Project[] = [
    {
        id: "customer-churn-prediction",
        title: "Customer Churn Prediction using EDA, Feature Engineering & ML",
        description: "This project aims to predict customer churn probability using Exploratory Data Analysis (EDA), Feature Engineering, and Machine Learning models.",
        longDescription: `
### Project Overview
This project aims to predict customer churn probability using Exploratory Data Analysis (EDA), Feature Engineering, and Machine Learning models.

Churn is a critical issue for businesses, as identifying at-risk customers allows for data-driven preventive actions.

{{OVERVIEW_IMAGE}}

### Dataset
The dataset contains customer information with variables related to:

{{DATASET_TABLE}}

{{ANALYSIS_OBJECTIVE}}

### Exploratory Data Analysis (EDA)
During the exploratory analysis, the following tasks were performed:

**1. Initial Data Inspection**
- Visualization of first rows, checking data types, null values, and dimensions.

{{INITIAL_DATA_IMAGE}}

**2. Target Variable Analysis (Churn)**
- Distribution of customers who churn vs those who stay.
- Visualization with bar charts.

{{TARGET_IMAGE}}

**3. Numerical Variables Analysis**
- Distribution of Tenure, MonthlyCharges, TotalCharges.
- Visualization with bar charts.
- Identification of outliers using boxplots.

{{NUMERICAL_DISTRIBUTION_IMAGE}}
{{SKEWNESS_IMAGE}}

**4. Categorical Variables Analysis**
- Churn distribution across categorical customer attributes: Gender, Education Level, Marital Status, and Income Category.

{{CATEGORICAL_IMAGE}}

**5. Correlation between Variables**
- Correlation matrix and heatmap to identify relevant relationships.

{{CORRELATION_IMAGE}}

### Data Cleaning & Feature Engineering
The following steps were taken to prepare the data:
- Conversion of columns to numeric types.
- Handling missing values.
- Encoding: Label Encoding and One-Hot Encoding for categorical variables.
- Scaling: Normalization of numeric variables.
- Splitting into Train and Test sets.

{{CLEANING_IMAGE}}

### Machine Learning Models
Several classification algorithms were trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

{{MODELS_IMAGE}}

### Model Evaluation
Models were evaluated using metrics such as:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC

{{EVALUATION_IMAGE}}

**📊 Model Performance Conclusion**

Among all evaluated models, **Gradient Boosting** achieved the best overall performance, with the highest:
- **Accuracy (0.9645)**
- **F1-score (0.8804)**
- **ROC-AUC (0.9885)**

This indicates that Gradient Boosting provides the strongest balance between correctly identifying churners (recall) and minimizing false positives (precision), making it the most reliable model for churn prediction in this dataset.

Random Forest also performed competitively but showed slightly lower recall and F1-score, while Logistic Regression underperformed, particularly in recall, indicating limitations in capturing complex nonlinear patterns.

{{FEATURE_IMPORTANCE_IMAGE}}

### General Conclusions & Next Steps

Among all evaluated models, Gradient Boosting achieved the highest F1-score (0.8804), indicating the best balance between precision and recall for churn prediction. This makes it the most reliable model for correctly identifying churners while minimizing both false positives and false negatives.

Random Forest showed competitive performance (F1 = 0.8508) but with lower recall, while Logistic Regression underperformed significantly (F1 = 0.6510), highlighting its limitations in capturing complex, non-linear customer behavior patterns.

Next steps include improving model interpretability using SHAP, optimizing classification thresholds based on business costs, enhancing feature engineering with temporal and interaction features, and deploying the model within a production-ready API for real-time churn prediction.


### Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
        `,
        tags: ["Python", "Scikit-learn", "EDA", "Machine Learning"],
        imageUrl: churnMain,
        kaggleUrl: "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
        githubUrl: "https://www.kaggle.com/code/frankchipana/customer-churn-prediction-eda-fe-ml",
        codeTitle: "Code (Kaggle)",
        overviewImage: churnProcess,
        analysisImage: churnAnalysis,
        initialDataImage: churnInitialData,
        targetAnalysisImage: churnTargetAnalysis,
        targetDistributionImage: churnDistribution,
        skewnessImage: churnSkewness,
        numericalDistributionImage: churnNumerical,
        categoricalAnalysisImage: churnCategorical,
        correlationImage: churnCorrelation,
        cleaningImage: churnCleaning,
        modelsImage: churnModels,
        evaluationImage: churnEvaluation,
        featureImportanceImage: churnFeatureImportance,
        datasetSchema: [
            { category: "Demographics", variables: "Gender, SeniorCitizen, Partner, Dependents" },
            { category: "Service Usage", variables: "Tenure, PhoneService, InternetService, etc." },
            { category: "Contract Info", variables: "Contract, PaperlessBilling, PaymentMethod" },
            { category: "Financial Charges", variables: "MonthlyCharges, TotalCharges" },
            { category: "Target Variable", variables: "Churn (Yes / No)" }
        ]
    },
    {
        id: "customer-segmentation",
        title: "Customer Segmentation Clustering",
        description: "Customer segmentation pipeline using unsupervised learning to uncover behavioral groups and drive targeted retention and campaign strategies.",
        longDescription: `
### Business Context
Marketing and CRM teams were using one-size-fits-all campaigns, resulting in low engagement and inefficient budget allocation.

### Project Goal
Identify meaningful customer segments based on behavior and value signals to support personalized communication, retention plans, and cross-sell opportunities.

### Dataset Scope
- Customer demographics and tenure data.
- Purchase frequency and monetary behavior.
- Product usage and channel interaction metrics.
- Campaign response history.

{{DATASET_HEAD_IMAGE}}

### Exploratory Data Analysis (EDA)
1. Compared customer behavior by parenthood status using pairwise distributions to see how Income, Recency, Customer_For, Age, and Spent differ across groups.
2. Checked density shifts and spread to identify segments with higher spend and longer tenure.
3. Built a correlation matrix to detect strongly related variables and potential redundancy in purchase channels.
4. Used the EDA insights to decide which features would drive clustering and which could be simplified.

{{METHODOLOGY_IMAGE}}
{{METHODOLOGY_IMAGE_2}}

### Feature Engineering
- Derived customer Age from Year_Birth.
- Built Spent as the total amount spent across product categories over two years.
- Created Living_With from Marital_Status to capture household living situation.
- Created Children as the total number of kids and teenagers in the household.
- Added Family_Size for clearer household segmentation.
- Added Is_Parent to indicate parenthood status.
- Simplified Education into three categories for cleaner grouping.
- Applied OneHotEncoder for categorical variables.
- Applied StandardScaler for numerical variables.
- Used PCA to reduce dimensionality and keep the most informative variance for clustering.
- Dropped redundant or low-signal features.

{{FEATURE_ENGINEERING_IMAGE}}

### Modeling Approach
- K-Means (primary model) with Elbow Method to select the optimal number of clusters.
- Agglomerative Clustering to explore hierarchical structure and improve interpretability using dendrograms.
- Best suited for small-to-medium cluster structures where relationships between groups matter.

{{MODELS_IMAGE}}

### Segment Profiles
After forming the clusters and reviewing their purchase behavior, I profiled each group to understand who the “star” customers are and which segments need more attention from the marketing team.
To do this, I focused on key drivers by cluster such as Income, Recency, Customer_For, Age, Spent, Children, Family_Size, and Is_Parent, plus a few high-signal product/channel metrics like Wines, Meat, NumWebPurchases, and NumStorePurchases. Based on these patterns, I summarized each segment and translated the insights into clear marketing priorities.

{{PROFILING_IMAGE}}
{{PROFILING_IMAGE_2}}

### Results
The analysis reveals that customer segmentation is primarily driven by spending behavior rather than demographic variables such as age or customer tenure. High-spending clusters are predominantly composed of customers with smaller household sizes and fewer children, while lower-spending clusters tend to include larger families. This suggests that family structure plays a significant role in purchasing capacity. Customer tenure shows no strong correlation with spending, indicating that loyalty is not solely time-dependent but behavior-driven.

### Business Impact
This segmentation enabled targeted campaigns aligned to spending capacity and household structure, improving marketing efficiency and reducing wasted budget.
It also provided a clear framework for prioritizing retention outreach and tailoring promotions by channel preference (web, catalog, store).
The final cluster definitions serve as a reusable foundation for personalization, recommendation, and lifecycle strategies.

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (K-Means, PCA, metrics)
- Matplotlib / Seaborn
- Jupyter Notebook
        `,
        tags: ["Python", "Clustering", "K-Means", "PCA", "Customer Analytics"],
        imageUrl: segmentationCardImage,
        githubUrl: "https://www.kaggle.com/code/frankchipana/clustering-customer-personality-analysis",
        codeTitle: "Code (Kaggle)",
        kaggleUrl: "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis",
        datasetTitle: "Kaggle Dataset",
        overviewImage: segmentationDatasetHead,
        analysisImage: segmentationMethodology,
        initialDataImage: segmentationMethodologySecond,
        numericalDistributionImage: segmentationFeatureEngineering,
        modelsImage: segmentationModelsCompared,
        modelsImageSecondary: segmentationModelingApproachSecond,
        modelsImageTertiary: segmentationModelingApproachThird,
        profilingImage: segmentationProfiling,
        profilingImageSecondary: segmentationProfilingSecond
    },
    {
        id: "credit-risk-analysis",
        title: "RAG Assistant Deployment (Portfolio Chatbot)",
        description: "End-to-end Retrieval-Augmented Generation project integrated into this portfolio, with a Python RAG backend and a React chatbot UI.",
        longDescription: `
### Project Scheme
{{PROJECT_SCHEME_IMAGE}}

### Business Context
I wanted my portfolio to do more than show static projects. The goal was to let visitors ask questions and get grounded answers from indexed documents in real time.

### Project Goal
Transform the initial local RAG prototype into a working web feature connected to the portfolio:
- Python agent for retrieval and generation.
- HTTP API layer to expose chat endpoints.
- Frontend chatbot integrated into the portfolio UI.

### Dataset Scope
- Course and technical PDFs stored as local documents.
- Documents split into chunks and embedded.
- Vector search over Azure Cosmos DB for context retrieval.

### Methodology
1. Refactored the original CLI-only agent to support both CLI and API execution modes.
2. Encapsulated the RAG pipeline into a reusable service layer (LLM + embeddings + retriever + prompt).
3. Exposed a lightweight HTTP API with endpoints for health, connect, reindex, and chat.
4. Implemented a floating chatbot UI in React and mounted it globally in the layout.
5. Added a “Try Chatbot” call-to-action that opens the chat widget from project cards and detail pages.
6. Hardened deployment behavior (fast port binding, CORS, and health checks) for production hosting.

### Feature Engineering
- Text chunking using recursive splitting.
- Embeddings generated with HuggingFace sentence-transformers.
- Vector fields and policies configured for Azure Cosmos DB vector search.
- Retrieval setup with top-k context injection into the prompt.

### Deployment & MLOps
- Containerized the backend with Docker for reproducible local testing and production builds.
- Built and pushed images with Google Cloud Build and deployed the service on Google Cloud Run.
- Configured runtime secrets and environment variables (Cosmos DB + Gemini) outside the codebase.
- Connected the GitHub Pages frontend to the Cloud Run backend through a single API base URL.
- Added SPA routing support for GitHub Pages (custom 404.html fallback).

### Validation
- Backend: /health checks, port binding verification, and request-level error handling.
- Frontend: production builds with Vite and smoke tests across routes and chat actions.
- Cross-origin testing: confirmed browser CORS behavior across GitHub Pages → Cloud Run.

### Results
- Portfolio now includes a functional chatbot connected to the RAG agent.
- The backend supports both CLI workflow and API workflow.
- Chat responses include contextual document sources for traceability.

### Business Impact
- Upgraded portfolio from static showcase to interactive technical demo.
- Demonstrates practical MLOps-style integration across backend and frontend.
- Improves recruiter and client engagement by showing deployed AI capability.

### Tech Stack
- Frontend: React, TypeScript, Tailwind CSS, Vite, TanStack Router, GitHub Pages
- Backend: Python, LangChain, Gemini API, HuggingFace Embeddings, Azure Cosmos DB (Vector Search)
- DevOps: Docker, Google Cloud Build, Google Cloud Run, environment-based configuration
        `,
        tags: ["Python", "RAG", "LangChain", "Azure Cosmos DB", "React Chatbot"],
        imageUrl: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
        projectSchemeImage: ragProjectScheme,
        githubUrl: "https://github.com/frankchip2023",
        codeTitle: "Code (GitHub)",
        kaggleUrl: "https://www.kaggle.com/frankchipana",
        datasetTitle: "Kaggle Profile",
        demoUrl: "#open-chat",
        demoTitle: "Try Chatbot"
    },
    {
        id: "ppe-detection",
        title: "PPE Detection (Computer Vision)",
        description: "Implemented a YOLO-based object detection model to identify Personal Protective Equipment (PPE) compliance in real-time video feeds, enhancing workplace safety.",
        longDescription: `
### Overview
The objective of this project was to design and implement a Computer Vision model capable of detecting Personal Protective Equipment (PPE) in real-world working environments using YOLOv8 by Ultralytics.

This system helps automatically identify whether workers are complying with safety regulations, such as wearing helmets, masks, and glasses, enabling real-time monitoring and preventive safety actions.

The model classifies the following categories:
0: Glasses
1: Head
2: Helmet
3: Mask worn incorrectly
4: No glasses
5: With mask
6: Without mask

{{OVERVIEW_IMAGE}}

### Tech Stack
- **Framework**: PyTorch / YOLOv8
- **Data**: Custom dataset annotated using CVAT.
- **Deployment**: Inference API via FastAPI.

### Model Training (YOLOv8 by Ultralytics)
The model was trained using YOLOv8, a state-of-the-art object detection architecture optimized for both speed and accuracy.

Training involved:
- Transfer learning from pretrained YOLOv8 weights
- Fine-tuning on the PPE dataset
- Optimization using stochastic gradient descent
- Monitoring of loss curves and convergence

Key training metrics included:
- Classification loss
- Bounding box regression loss
- Objectness confidence

This allowed the model to progressively improve detection performance across epochs.


### Model Evaluation

The trained model was evaluated using standard object detection metrics:
- **mAP (Mean Average Precision)**
- **Precision**
- **Recall**

Evaluation was performed on a separate validation set to measure generalization performance.

Visual inspections were also conducted by running predictions on unseen images to verify:
- Correct localization of PPE
- Accurate class prediction
- Robustness under different conditions

### Performance
- Achieved **mAP@0.5 of 0.92**.
- Runs in real-time (30 FPS) on a standard GPU.
        `,
        tags: ["Python", "YOLO", "OpenCV", "Deep Learning"],
        imageUrl: ppeMain,
        kaggleUrl: "https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2",
        datasetTitle: "Roboflow",
        githubUrl: "https://github.com/frankchip2023/EPP_Streamlit",
        demoUrl: "https://share.streamlit.io/?utm_source=streamlit&utm_medium=referral&utm_campaign=main&utm_content=-ss-streamlit-io-cloudpagehero",
        overviewImage: ppeOverview
    },
    {
        id: "sales-forecasting",
        title: "Sales Forecasting Model",
        description: "End-to-end time series forecasting to predict weekly retail demand and improve inventory, promotion planning, and stock availability.",
        longDescription: `
### Business Context
Retail teams needed reliable weekly sales forecasts to reduce stockouts, avoid overstock, and plan promotions with greater confidence.

### Project Goal
Build a robust forecasting pipeline that predicts future sales and compares multiple models to select the best performer for production use.

### Dataset Scope
- Weekly historical sales by product category and store.
- Calendar variables (month, week, holidays, special campaigns).
- External signals (price changes and promotional windows).

### Methodology
1. Data quality checks and missing value treatment.
2. Time-based feature engineering.
3. Exploratory analysis of trend, seasonality, and anomalies.
4. Model training and tuning.
5. Rolling backtesting and error analysis.

### Feature Engineering
- Lag features (t-1, t-2, t-4, t-8).
- Rolling statistics (mean, std, min/max windows).
- Calendar features (week of year, month, holiday flags).
- Promotion and pricing indicators.

### Models Compared
- Seasonal Naive (baseline).
- ARIMA / SARIMA.
- Prophet.
- Gradient Boosting Regressor for tabular time features.

### Validation Strategy
- Rolling-origin cross-validation to simulate real forecasting scenarios.
- Metrics: MAE, RMSE, and MAPE.
- Segment-level validation by category/store to detect underperforming slices.

### Results
- Prophet improved MAE by 15% vs baseline.
- Best configuration reduced forecast bias during holiday peaks.
- More stable predictions for low-volume categories after feature smoothing.

### Business Impact
- Better replenishment planning and lower risk of stockout periods.
- Improved promotion planning with clearer demand expectations.
- Framework ready for monthly retraining and dashboard integration.

### Tech Stack
- Python, Pandas, NumPy
- Statsmodels, Prophet, Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook
        `,
        tags: ["Python", "Time Series", "Forecasting", "Prophet", "ARIMA"],
        imageUrl: "https://images.unsplash.com/photo-1543286386-2e659306cd6c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        githubUrl: "https://github.com/frankchip2023",
        codeTitle: "Code (GitHub)",
        kaggleUrl: "https://www.kaggle.com/frankchipana",
        datasetTitle: "Kaggle Profile"
    },
    {
        id: "coming-soon",
        title: "Project Coming Soon",
        description: "Working on an exciting new data science project. Stay tuned for updates on predictive modeling and analysis.",
        longDescription: "More details coming soon!",
        tags: ["Data Science", "Machine Learning"],
        imageUrl: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        githubUrl: "#"
    }
];
