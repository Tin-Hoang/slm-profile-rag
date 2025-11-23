Welcome to the Profile Chatbot Documents Directory
===================================================

This is a sample document to demonstrate the document processing capabilities.

About This Chatbot
------------------

This chatbot uses Retrieval Augmented Generation (RAG) to answer questions about
a professional profile. It works by:

1. Processing documents (PDF, Word, HTML, text files)
2. Splitting them into semantic chunks
3. Creating vector embeddings
4. Storing in ChromaDB vector database
5. Retrieving relevant context for user queries
6. Generating answers using a local LLM (Ollama)

How to Use
----------

1. DELETE this sample file
2. Add your actual profile documents:
   - Resume in PDF or Word format
   - Project reports and case studies
   - LinkedIn profile (export as PDF or HTML)
   - Publications, certifications
   - Portfolio descriptions
   - Any other professional documents

3. Build the vector store:
   python -m src.build_vectorstore

4. Run the Streamlit app:
   streamlit run app.py

Supported File Types
--------------------

- PDF (.pdf)
- Microsoft Word (.docx, .doc)
- HTML (.html, .htm)
- Plain text (.txt)
- Markdown (.md)

Tips for Better Results
-----------------------

- Use well-structured documents with clear headings
- Include detailed information about your experience
- Add context about projects (challenges, solutions, results)
- Keep documents professional and relevant
- Update regularly as you gain new experience

Sample Profile Information
--------------------------

Name: John Doe
Title: Software Engineer & AI Enthusiast
Location: San Francisco, CA

Skills:
- Python, JavaScript, TypeScript
- Machine Learning & AI
- Web Development (React, Node.js)
- Cloud Platforms (AWS, GCP)
- Docker, Kubernetes

Experience:
- Senior Software Engineer at Tech Corp (2020-Present)
  * Led development of AI-powered features
  * Improved system performance by 40%
  * Mentored junior developers

- Software Engineer at StartupXYZ (2018-2020)
  * Built scalable microservices
  * Implemented CI/CD pipelines
  * Contributed to open-source projects

Education:
- M.S. Computer Science, Stanford University (2018)
- B.S. Computer Science, UC Berkeley (2016)

Projects:
- Open-source RAG Framework (2024)
  * Built modular RAG system for document Q&A
  * 500+ GitHub stars
  * Used by Fortune 500 companies

- AI Resume Analyzer (2023)
  * ML-powered resume screening tool
  * 90% accuracy in skill extraction
  * Reduced hiring time by 60%

Certifications:
- AWS Certified Solutions Architect
- Google Cloud Professional ML Engineer
- Certified Kubernetes Administrator (CKA)

Interests:
- Contributing to open-source AI projects
- Writing technical blog posts
- Speaking at tech conferences
- Mentoring aspiring engineers

---

REMEMBER: Replace this sample content with your actual profile documents!

