# 🚀 Landing Page Analyzer

An AI-powered tool that analyzes landing pages for usability, user engagement, and conversion optimization. Designed for marketers, founders, and non-designers to improve their web pages based on UX best practices and machine learning insights.

---

## 🧠 Features

- ✅ **Accepts multiple input types:**
  - Website URL
  - HTML snippet
  - Screenshot/image of a landing page

- 📊 **Generates UX & conversion score** based on:
  - Call-To-Action (CTA) count
  - Heading structure & hierarchy
  - Paragraph and list analysis
  - Testimonials and trust signals
  - Score prediction using Random Forest Regression

- 🧠 **AI Suggestions:**
  - Uses Gemini API to suggest improvements
  - Gives tips on structure, readability, CTA placement, and design

- 📸 **Visual Input Support:**
  - Extracts features from screenshots using OCR and layout heuristics *(experimental)*

- 🌐 **Frontend + Backend stack:**
  - Frontend: React + CSS
  - Backend: Flask (Python) + Gemini API + scoring model

---

## 🛠️ Tech Stack

| Layer     | Technology           |
|-----------|----------------------|
| Frontend  | React, CSS           |
| Backend   | Flask (Python)       |
| ML/AI     | Gemini API, Random   |
|           | Forest Regression    |
| Deployment| Vercel / Railway     |

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mansingh-04/prooback
cd prooback
