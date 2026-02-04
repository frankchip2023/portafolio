import React from 'react';
import { BookOpen, Code, Database, GraduationCap, Award, ExternalLink } from 'lucide-react';

const About: React.FC = () => {
    return (
        <section id="about" className="py-20 bg-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">About Me</h2>
                    <div className="w-20 h-1 bg-blue-600 mx-auto rounded-full"></div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">

                    {/* Column 1: Profile Photo (Left - 3 cols) */}
                    <div className="lg:col-span-3 flex justify-center lg:justify-start">
                        <div className="relative w-64 h-64 lg:w-full lg:h-auto lg:aspect-square rounded-2xl overflow-hidden shadow-xl transform rotate-3 hover:rotate-0 transition-all duration-300">
                            <img
                                src="/src/assets/images/profile.jpg"
                                alt="Frank Chipana"
                                className="w-full h-full object-cover"
                            />
                        </div>
                    </div>

                    {/* Column 2: Professional Summary & Skills (Middle - 5 cols) */}
                    <div className="lg:col-span-5">
                        <h3 className="text-2xl font-semibold text-gray-800 mb-6">
                            Professional Summary
                        </h3>
                        <p className="text-gray-600 mb-6 leading-relaxed">
                            Data Scientist and AI professional with strong experience leading and implementing advanced analytics and artificial intelligence solutions across mining, industrial, and commercial environments.
                        </p>
                        <p className="text-gray-600 mb-8 leading-relaxed">
                            Specialized in predictive modeling, time series analysis, computer vision, and demand elasticity. Proven track record of delivering measurable business impact through optimized maintenance planning, increased sales, and enhanced decision-making. Experienced in designing end-to-end data solutions, including Python-based ETL pipelines and ML models.
                        </p>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div className="p-4 bg-blue-50 rounded-xl">
                                <Code className="text-blue-600 mb-3" size={24} />
                                <h4 className="font-semibold text-gray-900 mb-1">Languages</h4>
                                <p className="text-xs text-gray-600">Python (Advanced), SQL, JavaScript</p>
                            </div>
                            <div className="p-4 bg-indigo-50 rounded-xl">
                                <Database className="text-indigo-600 mb-3" size={24} />
                                <h4 className="font-semibold text-gray-900 mb-1">Data & ML</h4>
                                <p className="text-xs text-gray-600">Pandas, NumPy, Scikit-learn, Power BI</p>
                            </div>
                            <div className="p-4 bg-purple-50 rounded-xl sm:col-span-2">
                                <BookOpen className="text-purple-600 mb-3" size={24} />
                                <h4 className="font-semibold text-gray-900 mb-1">Cloud & Soft Skills</h4>
                                <p className="text-xs text-gray-600">GCP (Vertex AI), Azure, Problem Solving, Communication</p>
                            </div>
                        </div>
                    </div>

                    {/* Column 3: Education & Certifications (Right - 4 cols) */}
                    <div className="lg:col-span-4">
                        <h3 className="text-2xl font-semibold text-gray-900 mb-6">Education & Certifications</h3>
                        <div className="relative border-l border-gray-200 ml-3 space-y-8">

                            <div className="ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-green-100 rounded-full -left-3 ring-8 ring-white">
                                    <Award size={14} className="text-green-600" />
                                </span>
                                <h3 className="mb-1 text-base font-semibold text-gray-900 group">
                                    <a
                                        href="https://www.credly.com/badges/8c45a14c-cfec-46bd-a825-e6c7c9df153f/public_url"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="hover:text-blue-600 transition-colors flex items-center gap-2"
                                    >
                                        Google Advanced Data Analytics
                                        <ExternalLink size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </a>
                                </h3>
                                <time className="block mb-1 text-xs font-normal leading-none text-gray-400">2026</time>
                                <p className="text-sm font-normal text-gray-500">Professional Certificate by Google</p>
                            </div>

                            <div className="ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full -left-3 ring-8 ring-white">
                                    <GraduationCap size={14} className="text-blue-600" />
                                </span>
                                <h3 className="mb-1 text-base font-semibold text-gray-900">Master’s in Artificial Intelligence</h3>
                                <time className="block mb-1 text-xs font-normal leading-none text-gray-400">2025 - 2027 (Expected)</time>
                                <p className="text-sm font-normal text-gray-500">National University of Engineering (Universidad Nacional de Ingeniería)</p>
                            </div>

                            <div className="ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full -left-3 ring-8 ring-white">
                                    <GraduationCap size={14} className="text-blue-600" />
                                </span>
                                <h3 className="mb-1 text-base font-semibold text-gray-900">Master’s in Data Science</h3>
                                <time className="block mb-1 text-xs font-normal leading-none text-gray-400">2023 - 2025</time>
                                <p className="text-sm font-normal text-gray-500">University of Applied Sciences (Universidad Privada de Ciencias Aplicadas)</p>
                            </div>

                            <div className="ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full -left-3 ring-8 ring-white">
                                    <GraduationCap size={14} className="text-blue-600" />
                                </span>
                                <h3 className="mb-1 text-base font-semibold text-gray-900">Big Data Management</h3>
                                <time className="block mb-1 text-xs font-normal leading-none text-gray-400">2024 - 2025</time>
                                <p className="text-sm font-normal text-gray-500">Polytechnic University of Catalonia (Universidad Politécnica de Catalunya)</p>
                            </div>

                            <div className="ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full -left-3 ring-8 ring-white">
                                    <GraduationCap size={14} className="text-blue-600" />
                                </span>
                                <h3 className="mb-1 text-base font-semibold text-gray-900">Agricultural Engineering</h3>
                                <time className="block mb-1 text-xs font-normal leading-none text-gray-400">2011 - 2016</time>
                                <p className="text-sm font-normal text-gray-500">National Agrarian University (Universidad Nacional Agraria la Molina)</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default About;
