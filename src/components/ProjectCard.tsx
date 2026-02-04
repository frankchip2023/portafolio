import React from 'react';
import { Github, ExternalLink, Database } from 'lucide-react';
import { Link } from 'react-router-dom';

interface ProjectCardProps {
    id: string;
    title: string;
    description: string;
    tags: string[];
    imageUrl: string;
    githubUrl?: string;
    kaggleUrl?: string; // Additional link for Kaggle
    demoUrl?: string;
    codeTitle?: string;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ id, title, description, tags, imageUrl, githubUrl, kaggleUrl, demoUrl, codeTitle }) => {
    return (
        <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300 flex flex-col h-full border border-gray-100 group">
            <Link to={`/project/${id}`} className="block relative h-48 overflow-hidden cursor-pointer">
                <img
                    src={imageUrl}
                    alt={title}
                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all flex items-center justify-center">
                    <span className="opacity-0 group-hover:opacity-100 bg-white text-gray-900 px-4 py-2 rounded-full font-medium transform translate-y-4 group-hover:translate-y-0 transition-all duration-300 shadow-lg">
                        View Case Study
                    </span>
                </div>
            </Link>

            <div className="p-6 flex flex-col flex-grow">
                <Link to={`/project/${id}`} className="block">
                    <h3 className="text-xl font-bold text-gray-900 mb-2 hover:text-blue-600 transition-colors">{title}</h3>
                </Link>
                <p className="text-gray-600 mb-4 text-sm leading-relaxed flex-grow">
                    {description}
                </p>

                <div className="flex flex-wrap gap-2 mb-6">
                    {tags.map((tag) => (
                        <span key={tag} className="px-3 py-1 bg-blue-50 text-blue-600 text-xs font-medium rounded-full">
                            {tag}
                        </span>
                    ))}
                </div>

                <div className="flex gap-4 pt-4 border-t border-gray-100 mt-auto">
                    {githubUrl && (
                        <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="flex items-center text-sm font-medium text-gray-700 hover:text-blue-600 transition-colors">
                            <Github size={18} className="mr-2" />
                            {codeTitle || "Code"}
                        </a>
                    )}
                    {kaggleUrl && (
                        <a href={kaggleUrl} target="_blank" rel="noopener noreferrer" className="flex items-center text-sm font-medium text-gray-700 hover:text-blue-600 transition-colors">
                            <Database size={18} className="mr-2" />
                            Kaggle
                        </a>
                    )}
                    {demoUrl && (
                        <a href={demoUrl} target="_blank" rel="noopener noreferrer" className="flex items-center text-sm font-medium text-gray-700 hover:text-blue-600 transition-colors">
                            <ExternalLink size={18} className="mr-2" />
                            View Project
                        </a>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ProjectCard;
