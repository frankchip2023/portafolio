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
import ppeOverview from '../assets/images/ppe_overview.png';
import ppeMain from '../assets/images/ppe_main.jpg';

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
    codeTitle?: string;
    overviewImage?: string; // Single image to be embedded in description
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

### Methodology
1. Data cleaning and outlier handling.
2. Feature scaling and transformation.
3. Exploratory analysis of customer behavior distributions.
4. Clustering model selection and tuning.
5. Segment profiling and business interpretation.

### Feature Engineering
- RFM-style features (Recency, Frequency, Monetary).
- Ratios such as average order value and engagement rate.
- Standardization with StandardScaler.
- PCA projection for visualization and noise reduction.

### Models Compared
- K-Means (primary model).
- Hierarchical clustering (benchmark).
- DBSCAN (noise-aware benchmark).

### Validation Strategy
- Elbow Method and Silhouette Score for cluster count selection.
- Cluster size balance checks to avoid impractical micro-segments.
- Stability analysis across random seeds and time windows.

### Segment Profiles (Example)
1. High Value Loyalists: high frequency, high spend, low churn risk.
2. Price Sensitive Occasionals: moderate frequency, promo-driven purchases.
3. At-Risk Customers: declining activity, low engagement, high churn probability.
4. New Growth Segment: recent acquisition with rising activity trend.

### Results
- Identified 4 actionable customer segments.
- Improved campaign targeting design with segment-specific messaging.
- Clear separation achieved with strong silhouette performance.

### Business Impact
- Better allocation of marketing spend by segment value.
- Improved retention strategy for at-risk cohorts.
- Foundation created for personalized recommendation workflows.

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (K-Means, PCA, metrics)
- Matplotlib / Seaborn
- Jupyter Notebook
        `,
        tags: ["Python", "Clustering", "K-Means", "PCA", "Customer Analytics"],
        imageUrl: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        githubUrl: "https://github.com/frankchip2023",
        codeTitle: "Code (GitHub)",
        kaggleUrl: "https://www.kaggle.com/frankchipana",
        datasetTitle: "Kaggle Profile"
    },
    {
        id: "credit-risk-analysis",
        title: "Credit Risk Analysis",
        description: "Credit risk modeling pipeline to estimate default probability and support safer, data-driven lending decisions.",
        longDescription: `
### Business Context
Credit teams need reliable default-risk estimates to approve loans faster while controlling losses and regulatory exposure.

### Project Goal
Build a classification model that predicts customer default probability and helps prioritize high-risk applications for manual review.

### Dataset Scope
- Applicant demographics and income features.
- Credit behavior indicators and payment history.
- Loan attributes (term, amount, interest profile).
- Historical default label as target variable.

### Methodology
1. Data quality checks and missing-value treatment.
2. Categorical encoding and numeric scaling.
3. Class imbalance handling.
4. Model training and hyperparameter tuning.
5. Threshold analysis aligned with business risk tolerance.

### Feature Engineering
- Debt-to-income and utilization ratios.
- Payment delay aggregates and behavior trend features.
- Encoding for high-cardinality categories.
- Optional balancing with SMOTE for minority class recall.

### Models Compared
- Logistic Regression (interpretable baseline).
- Random Forest.
- XGBoost / LightGBM.

### Validation Strategy
- Stratified train/validation/test split.
- Cross-validation with ROC-AUC, Precision, Recall, F1.
- Confusion matrix and PR curve for threshold selection.

### Results
- Gradient boosting family delivered the best ROC-AUC.
- Higher recall for defaulters than baseline logistic model.
- More stable ranking of risky applicants across folds.

### Business Impact
- Improved risk segmentation for underwriting.
- Better trade-off between approval volume and default risk.
- Reusable scoring pipeline for future monitoring and retraining.

### Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
- Matplotlib / Seaborn
- Jupyter Notebook
        `,
        tags: ["Python", "Credit Risk", "XGBoost", "LightGBM", "Classification"],
        imageUrl: "https://images.unsplash.com/photo-1563986768609-322da13575f3?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        githubUrl: "https://github.com/frankchip2023",
        codeTitle: "Code (GitHub)",
        kaggleUrl: "https://www.kaggle.com/frankchipana",
        datasetTitle: "Kaggle Profile"
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
        id: "coming-soon",
        title: "Project Coming Soon",
        description: "Working on an exciting new data science project. Stay tuned for updates on predictive modeling and analysis.",
        longDescription: "More details coming soon!",
        tags: ["Data Science", "Machine Learning"],
        imageUrl: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        githubUrl: "#"
    }
];
