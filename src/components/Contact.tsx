import React from 'react';
import { Mail, MapPin } from 'lucide-react';

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
                    </div>

                    {/* Contact Form */}
                    <form className="space-y-6 bg-gray-50 p-8 rounded-2xl shadow-sm border border-gray-100">
                        <div>
                            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                            <input
                                type="text"
                                id="name"
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="Your Name"
                            />
                        </div>
                        <div>
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                            <input
                                type="email"
                                id="email"
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="your@email.com"
                            />
                        </div>
                        <div>
                            <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">Message</label>
                            <textarea
                                id="message"
                                rows={4}
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="Your message..."
                            ></textarea>
                        </div>
                        <button
                            type="submit"
                            className="w-full bg-blue-600 text-white font-medium py-3 rounded-lg hover:bg-blue-700 transition-colors shadow-md hover:shadow-lg"
                        >
                            Send Message
                        </button>
                    </form>
                </div>
            </div>
        </section>
    );
};

export default Contact;
