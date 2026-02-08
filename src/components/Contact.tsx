import React from 'react';
import { Link } from '@tanstack/react-router';
import { Mail, MapPin, Phone, Download, Briefcase, Clock3 } from 'lucide-react';

const Contact: React.FC = () => {
    return (
        <section id="contact" className="py-20 bg-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Get In Touch</h2>
                    <div className="w-20 h-1 bg-blue-600 mx-auto rounded-full"></div>
                </div>

                <div className="grid md:grid-cols-2 gap-12">
                    {/* Contact Info */}
                    <div className="space-y-8">
                        <h3 className="text-2xl font-bold text-gray-900 mb-6">Let's Connect</h3>
                        <p className="text-gray-600 mb-8 leading-relaxed">
                            I'm currently open to new opportunities in Data Science and Analytics.
                            Whether you have a question or just want to say hi, I'll try my best to get back to you!
                        </p>

                        <div className="flex items-start space-x-4">
                            <div className="bg-blue-100 p-3 rounded-lg text-blue-600">
                                <Mail size={24} />
                            </div>
                            <div>
                                <h4 className="text-lg font-semibold text-gray-900">Email</h4>
                                <a href="mailto:frank.chipana@outlook.com" className="text-gray-600 hover:text-blue-600 transition-colors">
                                    frank.chipana@outlook.com
                                </a>
                            </div>
                        </div>

                        <div className="flex items-start space-x-4">
                            <div className="bg-blue-100 p-3 rounded-lg text-blue-600">
                                <MapPin size={24} />
                            </div>
                            <div>
                                <h4 className="text-lg font-semibold text-gray-900">Location</h4>
                                <p className="text-gray-600">Oslo, Norway</p>
                            </div>
                        </div>

                        <div className="flex items-start space-x-4">
                            <div className="bg-blue-100 p-3 rounded-lg text-blue-600">
                                <Phone size={24} />
                            </div>
                            <div>
                                <h4 className="text-lg font-semibold text-gray-900">Phone</h4>
                                <div className="text-gray-600">
                                    <a href="tel:+4745840940" className="hover:text-blue-600 transition-colors">
                                        +47 458 40 940
                                    </a>
                                    <span className="mx-2 text-gray-400">/</span>
                                    <a href="tel:+51997465203" className="hover:text-blue-600 transition-colors">
                                        +51 997 465 203
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-6 bg-gray-50 p-8 rounded-2xl shadow-sm border border-gray-100">
                        <h3 className="text-2xl font-bold text-gray-900">Work With Me</h3>
                        <p className="text-gray-600 leading-relaxed">
                            I collaborate on data science, machine learning, and analytics projects from discovery to deployment.
                        </p>

                        <div className="space-y-4">
                            <div className="flex items-start gap-3">
                                <div className="bg-blue-100 p-2 rounded-lg text-blue-600">
                                    <Briefcase size={18} />
                                </div>
                                <p className="text-sm text-gray-700">
                                    Available for full-time roles, consulting, and project-based collaborations.
                                </p>
                            </div>
                            <div className="flex items-start gap-3">
                                <div className="bg-blue-100 p-2 rounded-lg text-blue-600">
                                    <Clock3 size={18} />
                                </div>
                                <p className="text-sm text-gray-700">
                                    Typical response time: within 24 hours.
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                            <a
                                href="mailto:frank.chipana@outlook.com"
                                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 transition-colors"
                            >
                                <Mail size={16} />
                                Send Email
                            </a>
                            <Link
                                to="/cv"
                                className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-200 bg-white px-4 py-3 text-sm font-semibold text-blue-700 hover:bg-blue-50 transition-colors"
                            >
                                <Download size={16} />
                                View CV
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Contact;
