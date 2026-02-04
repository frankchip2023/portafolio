import React, { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Github, Database, Play } from 'lucide-react';
import { projects } from '../data/projects';

const ProjectDetail: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const project = projects.find(p => p.id === id);

    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    if (!project) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="text-center">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Project not found</h2>
                    <Link to="/" className="text-blue-600 hover:text-blue-700 font-medium flex items-center justify-center gap-2">
                        <ArrowLeft size={20} />
                        Back to Portfolio
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <article className="py-20 bg-white">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                <Link to="/" className="inline-flex items-center text-gray-600 hover:text-blue-600 transition-colors mb-8 group">
                    <ArrowLeft size={20} className="mr-2 group-hover:-translate-x-1 transition-transform" />
                    Back to Portfolio
                </Link>

                <div className="mb-10 rounded-2xl overflow-hidden shadow-2xl">
                    <img
                        src={project.imageUrl}
                        alt={project.title}
                        className="w-full h-64 md:h-96 object-cover"
                    />
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                    {project.tags.map((tag) => (
                        <span key={tag} className="px-3 py-1 bg-blue-100 text-blue-700 text-sm font-medium rounded-full">
                            {tag}
                        </span>
                    ))}
                </div>

                <h1 className="text-4xl font-bold text-gray-900 mb-6">{project.title}</h1>

                <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                    {project.description}
                </p>

                {/* Links */}
                <div className="flex flex-wrap gap-4 mb-12 pb-12 border-b border-gray-100">
                    {project.githubUrl && project.githubUrl !== '#' && (
                        <a
                            href={project.githubUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center px-6 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
                        >
                            <Github size={20} className="mr-2" />
                            {project.codeTitle || "View Code"}
                        </a>
                    )}
                    {project.kaggleUrl && project.kaggleUrl !== '#' && (
                        <a
                            href={project.kaggleUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center px-6 py-3 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                        >
                            <Database size={20} className="mr-2" />
                            {project.datasetTitle || "Dataset / Kaggle"}
                        </a>
                    )}
                    {project.demoUrl && project.demoUrl !== '#' && (
                        <a
                            href={project.demoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                        >
                            <Play size={20} className="mr-2" />
                            Live Demo
                        </a>
                    )}
                </div>

                {/* Detailed Content */}
                {project.longDescription && (
                    <div className="prose prose-lg prose-blue max-w-none text-gray-700 leading-relaxed">
                        {project.longDescription.split('\n').map((line, index) => {
                            if (line.trim() === '{{OVERVIEW_IMAGE}}' && project.overviewImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.overviewImage}
                                            alt="Project Overview Process"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                    </div>
                                );
                            }
                            if (line.trim() === '{{DATASET_TABLE}}' && project.datasetSchema) {
                                return (
                                    <div key={index} className="my-8 overflow-hidden rounded-xl border border-gray-200 shadow-sm">
                                        <table className="min-w-full divide-y divide-gray-200">
                                            <thead className="bg-gray-50">
                                                <tr>
                                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Variables</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200">
                                                {project.datasetSchema.map((item, i) => (
                                                    <tr key={i} className="hover:bg-gray-50 transition-colors">
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 border-r border-gray-100">
                                                            {item.category}
                                                        </td>
                                                        <td className="px-6 py-4 text-sm text-gray-600">
                                                            {item.variables}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{ANALYSIS_OBJECTIVE}}') {
                                return (
                                    <div key={index} className="flex flex-col lg:flex-row gap-8 items-center my-12">
                                        <div className="flex-1">
                                            <h3 className="text-2xl font-bold text-gray-900 mb-6 pb-2 border-b border-gray-200">Analysis Objective</h3>
                                            <p className="mb-4">The main objective is to build a model that allows:</p>
                                            <ul className="list-disc pl-5 space-y-2 mb-4 marker:text-blue-500">
                                                <li>Classifying whether a client will churn or not.</li>
                                                <li>Identifying which variables have the most influence on churn.</li>
                                                <li>Comparing different Machine Learning algorithms.</li>
                                            </ul>
                                        </div>
                                        {project.analysisImage && (
                                            <div className="flex-1">
                                                <img
                                                    src={project.analysisImage}
                                                    alt="Analysis Objectives Visual"
                                                    className="w-full h-auto rounded-xl shadow-lg border border-gray-100"
                                                />
                                            </div>
                                        )}
                                    </div>
                                );
                            }
                            if (line.trim() === '{{INITIAL_DATA_IMAGE}}' && project.initialDataImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.initialDataImage}
                                            alt="Initial Data Inspection Head"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Dataset Head View</p>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{TARGET_IMAGE}}' && (project.targetAnalysisImage || project.targetDistributionImage)) {
                                return (
                                    <div key={index} className="my-8 flex flex-col md:flex-row gap-6 items-stretch">
                                        {project.targetDistributionImage && (
                                            <div className="w-full md:w-1/3 rounded-xl overflow-hidden shadow-lg border border-gray-100 flex flex-col">
                                                <img
                                                    src={project.targetDistributionImage}
                                                    alt="Churn Distribution"
                                                    className="w-full flex-grow object-fill bg-white"
                                                />
                                                <p className="text-center text-sm text-gray-500 py-2 bg-gray-50 flex-shrink-0">Overall Churn Distribution</p>
                                            </div>
                                        )}
                                        {project.targetAnalysisImage && (
                                            <div className="w-full md:w-2/3 rounded-xl overflow-hidden shadow-lg border border-gray-100 flex flex-col">
                                                <img
                                                    src={project.targetAnalysisImage}
                                                    alt="Target Variable Analysis"
                                                    className="w-full h-auto object-contain bg-white"
                                                />
                                                <p className="text-center text-sm text-gray-500 py-2 bg-gray-50 flex-shrink-0">Distribution by Feature</p>
                                            </div>
                                        )}
                                    </div>
                                );
                            }
                            if (line.trim() === '{{NUMERICAL_DISTRIBUTION_IMAGE}}' && project.numericalDistributionImage) {
                                return (
                                    <div key={index} className="my-12 flex flex-col lg:flex-row gap-8 items-stretch">
                                        <div className="lg:w-3/4 rounded-xl overflow-hidden shadow-lg border border-gray-100 flex flex-col">
                                            <img
                                                src={project.numericalDistributionImage}
                                                alt="Numerical Feature Distributions"
                                                className="w-full h-full object-fill bg-white"
                                            />
                                            <p className="text-center text-sm text-gray-500 py-2 bg-gray-50 flex-shrink-0">Distribution & Boxplots</p>
                                        </div>
                                        <div className="lg:w-1/4 space-y-3 text-gray-700 text-sm leading-relaxed">
                                            <p>
                                                <strong>Customer_Age</strong> shows an approximately normal distribution centered around 45–50 years, with no significant skewness or extreme outliers, indicating a stable and homogeneous customer population suitable for direct use in modeling.
                                            </p>
                                            <p>
                                                <strong>Months_on_book</strong> presents a fairly symmetric distribution around 36 months with limited extreme values, suggesting consistent customer tenure and potential predictive relevance for churn behavior.
                                            </p>
                                            <p>
                                                <strong>Credit_Limit</strong> is highly right-skewed with large dispersion and many outliers, reflecting strong heterogeneity in customers’ purchasing power. This variable requires transformation (e.g., logarithmic scaling) before being used in predictive models.
                                            </p>
                                            <p>
                                                <strong>Total_Revolving_Bal</strong> exhibits an asymmetric distribution with concentration at mid-range values and some near-zero balances, indicating different credit usage patterns among customers and making it a key financial feature for churn analysis.
                                            </p>
                                        </div>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{SKEWNESS_IMAGE}}' && project.skewnessImage) {
                                return (
                                    <div key={index} className="my-8 max-w-xl mx-auto rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.skewnessImage}
                                            alt="Skewness Analysis"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Skewness for Dataset Features</p>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{CATEGORICAL_IMAGE}}' && project.categoricalAnalysisImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.categoricalAnalysisImage}
                                            alt="Categorical Variables Analysis"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Distribution of Categorical Variables</p>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{CORRELATION_IMAGE}}' && project.correlationImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.correlationImage}
                                            alt="Correlation Matrix"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Correlation Heatmap</p>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{CLEANING_IMAGE}}' && project.cleaningImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.cleaningImage}
                                            alt="Data Cleaning Pipeline"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                    </div>
                                );
                            }
                            if (line.trim() === '{{MODELS_IMAGE}}' && project.modelsImage) {
                                return (
                                    <div key={index} className="my-8 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.modelsImage}
                                            alt="Machine Learning Models"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Algorithms Compared</p>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{FEATURE_IMPORTANCE_IMAGE}}' && project.featureImportanceImage) {
                                return (
                                    <div key={index} className="my-12">
                                        <h3 className="text-2xl font-bold text-gray-900 mb-6">🧠 Feature Importance (Gradient Boosting)</h3>
                                        <div className="flex flex-col lg:flex-row gap-8 items-center">
                                            <div className="lg:w-2/3 rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                                <img
                                                    src={project.featureImportanceImage}
                                                    alt="Feature Importance Plot"
                                                    className="w-full h-auto object-contain bg-white"
                                                />
                                                <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Top 10 Most Important Features</p>
                                            </div>
                                            <div className="lg:w-1/3 space-y-4 text-gray-700 text-base leading-relaxed">
                                                <p>
                                                    The feature importance analysis reveals that <strong>customer transactional behavior</strong> is the dominant driver of churn.
                                                </p>
                                                <p>
                                                    These variables reflect how actively customers use their credit cards and how their behavior changes over time, confirming that churn is primarily driven by <strong>engagement and spending patterns</strong> rather than demographic attributes.
                                                </p>
                                                <p>
                                                    Demographic features such as <em>Customer_Age</em>, <em>Dependent_count</em>, and <em>Months_on_book</em> show relatively low importance, suggesting that churn is less dependent on who the customer is and more on how the customer behaves.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                );
                            }
                            if (line.trim() === '{{EVALUATION_IMAGE}}' && project.evaluationImage) {
                                return (
                                    <div key={index} className="my-8 max-w-3xl mx-auto rounded-xl overflow-hidden shadow-lg border border-gray-100">
                                        <img
                                            src={project.evaluationImage}
                                            alt="Model Evaluation Metrics"
                                            className="w-full h-auto object-contain bg-white"
                                        />
                                        <p className="text-center text-sm text-gray-500 py-2 bg-gray-50">Combined Metric Table</p>
                                    </div>
                                );
                            }
                            if (line.startsWith('### ')) return <h3 key={index} className="text-2xl font-bold text-gray-900 mt-10 mb-5 pb-2 border-b border-gray-200">{line.replace('### ', '')}</h3>;
                            if (line.startsWith('**') && line.endsWith('**')) return <strong key={index} className="block text-xl text-gray-800 mt-8 mb-3">{line.replace(/\*\*/g, '')}</strong>;
                            if (line.startsWith('**')) {
                                return <p key={index} className="mb-4" dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></p>
                            }
                            if (line.startsWith('- ')) {
                                return <li key={index} className="ml-6 mb-2 list-disc pl-2 text-gray-700 marker:text-blue-500" dangerouslySetInnerHTML={{ __html: line.replace('- ', '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></li>;
                            }
                            if (line.match(/^\d+\. /)) {
                                return (
                                    <li
                                        key={index}
                                        className="ml-6 mb-2 list-disc pl-2 text-gray-700 marker:text-blue-500"
                                    >
                                        {line.replace(/^\d+\. /, '')}
                                    </li>
                                );
                            }
                            if (line.trim() === '') return null;
                            return <p key={index} className="mb-4">{line}</p>;
                        })}
                    </div>
                )}
            </div>
        </article>
    );
};

export default ProjectDetail;
