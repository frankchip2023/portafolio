import React from 'react';
import { ArrowRight } from 'lucide-react';

const Hero: React.FC = () => {
    return (
        <section id="home" className="py-24 flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 mt-16">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center">
                <div className="mb-10">
                    <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 tracking-tight">
                        Frank Chipana <br />
                        <span className="text-blue-600 text-4xl md:text-6xl">Data Scientist</span>
                    </h1>
                    <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                        Specialized in predictive modeling, computer vision, and building end-to-end data solutions. Based in Oslo, Norway.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <a
                            href="#projects"
                            className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg transition-all shadow-lg hover:shadow-xl"
                        >
                            View Projects
                            <ArrowRight className="ml-2 -mr-1" size={20} />
                        </a>
                        <a
                            href="#contact"
                            className="inline-flex items-center justify-center px-8 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 md:py-4 md:text-lg transition-all"
                        >
                            Contact Me
                        </a>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Hero;
