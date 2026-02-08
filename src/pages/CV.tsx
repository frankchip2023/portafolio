import React from 'react';
import { Link } from '@tanstack/react-router';
import { ArrowLeft, Download, ExternalLink } from 'lucide-react';

const CV: React.FC = () => {
  const pdfUrl = `${import.meta.env.BASE_URL}cv-frank-chipana.pdf`;

  return (
    <section className="min-h-screen bg-white py-20">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <Link
          to="/"
          className="inline-flex items-center text-gray-600 hover:text-blue-600 transition-colors mb-6 group"
        >
          <ArrowLeft size={20} className="mr-2 group-hover:-translate-x-1 transition-transform" />
          Back to Portfolio
        </Link>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between mb-6">
          <div>
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900">Curriculum Vitae (PDF)</h1>
            <p className="text-gray-600 mt-2">
              Preview the document below. You can download it or open it in a new tab.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-sm font-semibold text-gray-800 hover:bg-gray-50 transition-colors"
            >
              <ExternalLink size={16} />
              Open in New Tab
            </a>
            <a
              href={pdfUrl}
              download
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 transition-colors"
            >
              <Download size={16} />
              Download PDF
            </a>
          </div>
        </div>

        <div className="rounded-2xl border border-gray-200 shadow-sm overflow-hidden bg-gray-50">
          <iframe
            title="Frank Chipana CV"
            src={pdfUrl}
            className="h-[80vh] w-full"
          />
        </div>

        <p className="text-sm text-gray-500 mt-4">
          If the PDF preview does not load in your browser, use the buttons above to open or download it.
        </p>
      </div>
    </section>
  );
};

export default CV;

