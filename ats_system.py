import autogen
import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import PyPDF2
from docx import Document
import pandas as pd
import requests
from pathlib import Path
import time
import hashlib
import re
import io
import base64
from difflib import SequenceMatcher
import numpy as np

# --- New Imports for PDF Export ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
# --- End New Imports ---

# Ensure 'coding' directory exists for dummy files
os.makedirs("coding", exist_ok=True)

# ===== FREE LLM CONFIGURATION OPTIONS =====

def get_free_llm_config(provider: str, api_key: str = ""):
    """Configure different free LLM providers for AutoGen"""
    
    if provider == "groq":
        # Groq - Very fast, free tier with good limits
        return [{
            "model": "llama3-8b-8192",  # Fast Llama 3 model
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "api_type": "openai",
        }]
    
    elif provider == "deepseek":
        # DeepSeek - Very generous free tier
        return [{
            "model": "deepseek-chat",
            "api_key": api_key,
            "base_url": "https://api.deepseek.com/v1",
            "api_type": "openai",
        }]
    
    elif provider == "together":
        # Together AI - Good free tier
        return [{
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "api_key": api_key,
            "base_url": "https://api.together.xyz/v1",
            "api_type": "openai",
        }]
    
    elif provider == "local_ollama":
        # Local Ollama - Completely free but requires local setup
        return [{
            "model": "llama3.2:3b",  # Smaller, faster model
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "api_type": "openai",
        }]
    
    else:
        return None

def test_llm_connection(config_list):
    """Test if the LLM configuration works"""
    try:
        test_agent = autogen.AssistantAgent(
            name="test_agent",
            system_message="Respond with 'OK' only.",
            llm_config={"config_list": config_list, "timeout": 15}
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        
        chat_result = user_proxy.initiate_chat(
            test_agent, 
            message="Test", 
            max_turns=1
        )
        
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

# Enhanced text normalization for consistent processing across formats
def normalize_text(text: str) -> str:
    """Normalize text to ensure consistency across different file formats"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    
    # Remove common formatting artifacts
    text = re.sub(r'[^\w\s@.\-+#():/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def advanced_skill_matching(resume_skills: List[str], required_skills: List[str]) -> np.ndarray:
    """Advanced skill matching with fuzzy matching and synonym detection"""
    if not resume_skills or not required_skills:
        return np.zeros((max(1, len(required_skills)), max(1, len(resume_skills))))
    
    # Create similarity matrix
    match_matrix = np.zeros((len(required_skills), len(resume_skills)))
    
    for i, req_skill in enumerate(required_skills):
        for j, res_skill in enumerate(resume_skills):
            # Multiple matching strategies
            similarity_scores = []
            
            # 1. Exact match
            if req_skill.lower().strip() == res_skill.lower().strip():
                similarity_scores.append(1.0)
            
            # 2. Substring match
            if req_skill.lower() in res_skill.lower() or res_skill.lower() in req_skill.lower():
                similarity_scores.append(0.8)
            
            # 3. Fuzzy string similarity
            similarity_scores.append(calculate_text_similarity(req_skill, res_skill))
            
            # 4. Word overlap
            req_words = set(req_skill.lower().split())
            res_words = set(res_skill.lower().split())
            if req_words and res_words:
                overlap = len(req_words.intersection(res_words)) / len(req_words.union(res_words))
                similarity_scores.append(overlap)
            
            # Take the maximum similarity score
            match_matrix[i, j] = max(similarity_scores) if similarity_scores else 0
    
    return match_matrix

# Optimized caching for file parsing
@st.cache_data
def parse_resume_text_from_upload(uploaded_file_bytes, file_extension):
    """Parses bytes content of an uploaded file to extract text with normalization."""
    try:
        text = ""
        if file_extension == 'pdf':
            reader = PyPDF2.PdfReader(uploaded_file_bytes)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif file_extension == 'docx':
            temp_path = Path("temp_doc.docx")
            temp_path.write_bytes(uploaded_file_bytes.getvalue())
            doc = Document(str(temp_path))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            temp_path.unlink()
        elif file_extension == 'txt':
            text = uploaded_file_bytes.getvalue().decode('utf-8')
        
        # Normalize text for consistency
        return normalize_text(text)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return ""

# Enhanced Database with Assignment Requirements
class ResumeDatabase:
    def __init__(self, db_path="resume_ats.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop and recreate tables to ensure proper schema
        cursor.execute('DROP TABLE IF EXISTS scores')
        cursor.execute('DROP TABLE IF EXISTS resumes')
        cursor.execute('DROP TABLE IF EXISTS job_descriptions')
        cursor.execute('DROP TABLE IF EXISTS job_templates')
        cursor.execute('DROP TABLE IF EXISTS user_sessions')
        cursor.execute('DROP TABLE IF EXISTS open_jobs')
        
        # Resumes table with normalized content
        cursor.execute('''
            CREATE TABLE resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                normalized_content TEXT,
                processed_data TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT UNIQUE,
                file_size INTEGER DEFAULT 0,
                content_hash TEXT
            )
        ''')
        
        # Enhanced scores table with consistency tracking
        cursor.execute('''
            CREATE TABLE scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                job_id INTEGER,
                total_score REAL,
                skills_score REAL,
                experience_score REAL,
                education_score REAL,
                format_score REAL,
                keyword_score REAL,
                score_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                score_hash TEXT,
                content_hash TEXT,
                job_hash TEXT,
                confidence_interval REAL DEFAULT 0.95,
                processing_time REAL DEFAULT 0,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Job descriptions table
        cursor.execute('''
            CREATE TABLE job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT,
                description TEXT NOT NULL,
                normalized_description TEXT,
                requirements TEXT,
                industry TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT
            )
        ''')
        
        # RAG Knowledge Base - Industry requirements
        cursor.execute('''
            CREATE TABLE job_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                industry TEXT,
                role_type TEXT,
                requirements TEXT,
                keywords TEXT,
                best_practices TEXT
            )
        ''')
        
        # User sessions for session management
        cursor.execute('''
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Open jobs database for recommendations
        cursor.execute('''
            CREATE TABLE open_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                industry TEXT,
                location TEXT,
                requirements TEXT,
                skills_needed TEXT,
                salary_range TEXT,
                posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Populate sample data for RAG and job recommendations
        self.populate_sample_data(cursor)
        
        conn.commit()
        conn.close()
    
    def populate_sample_data(self, cursor):
        """Populate sample data for RAG and job recommendations"""
        
        # Sample job templates for RAG
        job_templates = [
            ("Technology", "Software Engineer", "Programming languages, algorithms, data structures", 
             "Python,Java,JavaScript,React,Node.js,SQL,Git,Agile", 
             "Include specific technologies, showcase projects, quantify achievements"),
            ("Technology", "Data Scientist", "Statistics, machine learning, data analysis", 
             "Python,R,SQL,TensorFlow,Pandas,Scikit-learn,Tableau,AWS", 
             "Highlight analytical projects, include metrics, show domain expertise"),
            ("Marketing", "Digital Marketing", "SEO, SEM, social media, analytics", 
             "Google Analytics,Facebook Ads,SEO,SEM,Content Marketing,A/B Testing", 
             "Show campaign results, include conversion metrics, demonstrate ROI"),
            ("Finance", "Financial Analyst", "Financial modeling, analysis, reporting", 
             "Excel,SQL,Bloomberg,Financial Modeling,Valuation,Risk Analysis", 
             "Include CFA or similar certifications, show analytical achievements"),
            ("Healthcare", "Nurse", "Patient care, medical procedures, documentation", 
             "Patient Care,Medical Records,CPR,IV Therapy,Medication Administration", 
             "Include certifications, highlight patient outcomes, show continuing education")
        ]
        
        for template in job_templates:
            cursor.execute('''
                INSERT INTO job_templates (industry, role_type, requirements, keywords, best_practices)
                VALUES (?, ?, ?, ?, ?)
            ''', template)
        
        # Sample open jobs for recommendations
        open_jobs = [
            ("Senior Software Engineer", "TechCorp Inc", "Technology", "San Francisco, CA",
             "5+ years experience, Python, React, AWS", "Python,React,AWS,Docker,Kubernetes", "$120,000 - $180,000"),
            ("Data Scientist", "DataViz Solutions", "Technology", "Remote",
             "3+ years ML experience, Python, SQL", "Python,SQL,TensorFlow,Pandas,Statistics", "$95,000 - $140,000"),
            ("Frontend Developer", "WebDesign Pro", "Technology", "New York, NY",
             "React, JavaScript, CSS, responsive design", "React,JavaScript,CSS,HTML,Git", "$80,000 - $120,000"),
            ("Marketing Manager", "Growth Dynamics", "Marketing", "Austin, TX",
             "Digital marketing, campaign management", "Google Analytics,SEM,Content Marketing,A/B Testing", "$70,000 - $95,000"),
            ("DevOps Engineer", "CloudFirst", "Technology", "Seattle, WA",
             "Docker, Kubernetes, AWS, CI/CD", "Docker,Kubernetes,AWS,Jenkins,Terraform", "$110,000 - $160,000")
        ]
        
        for job in open_jobs:
            cursor.execute('''
                INSERT INTO open_jobs (title, company, industry, location, requirements, skills_needed, salary_range)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', job)
    
    def get_content_hash(self, content: str) -> str:
        """Generate hash based on normalized content for consistency checking"""
        normalized = normalize_text(content)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_resume_hash(self, content: str) -> str:
        """Generate hash for file-based consistency checking"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_resume(self, filename: str, content: str, processed_data: Dict, file_size: int = 0) -> int:
        """Save resume to database with normalized content for consistency"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        normalized_content = normalize_text(content)
        file_hash = self.get_resume_hash(content)
        content_hash = self.get_content_hash(content)
        
        cursor.execute('''
            INSERT OR REPLACE INTO resumes (filename, content, normalized_content, processed_data, file_hash, file_size, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, content, normalized_content, json.dumps(processed_data), file_hash, file_size, content_hash))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return resume_id
    
    def check_existing_score(self, content_hash: str, job_hash: str) -> Optional[Dict]:
        """Check if a score already exists for this content-job combination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT total_score, skills_score, experience_score, education_score, 
                   format_score, keyword_score, processing_time, confidence_interval
            FROM scores 
            WHERE content_hash = ? AND job_hash = ?
            ORDER BY score_date DESC LIMIT 1
        ''', (content_hash, job_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_score': result[0],
                'skills_score': result[1],
                'experience_score': result[2],
                'education_score': result[3],
                'format_score': result[4],
                'keyword_score': result[5],
                'processing_time': result[6],
                'confidence_level': result[7],
                'explanation': "Score retrieved from cache for consistency",
                'benchmark_comparison': "Consistent with previous analysis"
            }
        return None
    
    def save_scores(self, resume_id: int, scores: Dict, content_hash: str, job_hash: str, processing_time: float = 0) -> None:
        """Save scores with content and job hashes for consistency tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        score_hash = hashlib.md5(json.dumps(scores, sort_keys=True).encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO scores (resume_id, total_score, skills_score, experience_score, 
                              education_score, format_score, keyword_score, score_hash, 
                              content_hash, job_hash, confidence_interval, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (resume_id, scores['total_score'], scores['skills_score'], 
              scores['experience_score'], scores['education_score'], 
              scores['format_score'], scores['keyword_score'], score_hash,
              content_hash, job_hash, 0.95, processing_time))
        
        conn.commit()
        conn.close()
    
    def get_relevant_jobs(self, resume_skills: List[str], top_n: int = 5) -> List[Dict]:
        """RAG-based job recommendations based on resume skills"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all active jobs
        cursor.execute('''
            SELECT title, company, industry, location, requirements, skills_needed, salary_range
            FROM open_jobs WHERE is_active = 1
        ''')
        
        jobs = cursor.fetchall()
        conn.close()
        
        # Score jobs based on advanced skill matching
        job_scores = []
        for job in jobs:
            title, company, industry, location, requirements, skills_needed, salary = job
            job_skills = [skill.strip() for skill in skills_needed.split(',')]
            
            # Use advanced skill matching
            match_matrix = advanced_skill_matching(resume_skills, job_skills)
            match_score = np.mean(np.max(match_matrix, axis=1)) * 100 if match_matrix.size > 0 else 0
            
            job_scores.append({
                'title': title,
                'company': company,
                'industry': industry,
                'location': location,
                'requirements': requirements,
                'skills_needed': skills_needed,
                'salary_range': salary,
                'match_score': match_score
            })
        
        # Return top N jobs sorted by match score
        return sorted(job_scores, key=lambda x: x['match_score'], reverse=True)[:top_n]
    
    def get_industry_best_practices(self, industry: str) -> Dict:
        """RAG retrieval for industry-specific best practices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role_type, requirements, keywords, best_practices
            FROM job_templates WHERE industry = ?
        ''', (industry,))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return {
                'industry': industry,
                'roles': [{'role': r[0], 'requirements': r[1], 'keywords': r[2], 'best_practices': r[3]} for r in results]
            }
        return {}

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM responses"""
    try:
        # Try to find JSON block
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            
            # Clean up common JSON formatting issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'\n\s*', ' ', json_str)  # Replace newlines with spaces
            
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}. Using fallback extraction.")
    except Exception as e:
        st.warning(f"Response parsing error: {e}")
    
    return {}

# Enhanced Resume Processing Agent with consistent extraction
class ResumeProcessingAgent:
    def __init__(self, llm_config, db: ResumeDatabase):
        self.llm_config = llm_config
        self.db = db
        self.agent = autogen.AssistantAgent(
            name="ResumeProcessor",
            system_message="""You are a resume processing specialist. Extract structured information from resume text with ABSOLUTE CONSISTENCY.

Your tasks - EXTRACT EXACTLY THE SAME INFORMATION regardless of file format:
1. Extract personal information (name, email, phone, location)
2. Identify all skills mentioned (be comprehensive and consistent)
3. Parse work experience with companies, roles, duration, descriptions
4. Extract education details (institutions, degrees, fields, years)
5. Find certifications and licenses
6. Identify the likely industry/field

CRITICAL CONSISTENCY RULES:
- Always extract skills in the same format (normalize case, remove duplicates)
- Use consistent date formats (YYYY-YYYY or YYYY-Present)
- Standardize company names and job titles
- Always return the same industry classification for the same content

CRITICAL: Return ONLY valid JSON in this EXACT format - no additional text:
{
    "personal_info": {"name": "John Doe", "email": "john@email.com", "phone": "123-456-7890", "location": "City, State"},
    "skills": ["Python", "JavaScript", "SQL", "Machine Learning"],
    "experience": [{"company": "Company Name", "role": "Job Title", "duration": "2020-2023", "description": "Brief description"}],
    "education": [{"institution": "University Name", "degree": "Bachelor", "field": "Computer Science", "year": "2020"}],
    "certifications": ["AWS Certified", "Google Cloud"],
    "industry": "Technology"
}

CONSISTENCY IS CRITICAL: Same resume content must produce identical results regardless of format.
""",
            llm_config={"config_list": llm_config, "timeout": 30, "temperature": 0.0}  # Zero temperature for consistency
        )
    
    def process_resume(self, resume_text: str) -> Dict[str, Any]:
        """Process resume text and extract structured information with consistency"""
        # Normalize text first
        normalized_text = normalize_text(resume_text)
        
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        
        message = f"""
Extract structured information from this resume text. Use ZERO temperature and be absolutely consistent.
Return only the JSON structure with no additional text.

RESUME TEXT (normalized):
{normalized_text[:3000]}
"""
        
        try:
            chat_result = user_proxy.initiate_chat(self.agent, message=message, max_turns=1)
            response = chat_result.chat_history[-1]["content"]
            
            # Use robust JSON extraction
            extracted_data = extract_json_from_response(response)
            if extracted_data:
                # Post-process for consistency
                extracted_data = self._normalize_extracted_data(extracted_data)
                return extracted_data
                
        except Exception as e:
            st.error(f"Resume processing error: {e}")
        
        # Fallback structure
        return {
            "personal_info": {"name": "Not found", "email": "", "phone": "", "location": ""},
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "industry": "General"
        }
    
    def _normalize_extracted_data(self, data: Dict) -> Dict:
        """Normalize extracted data for consistency"""
        # Normalize skills (remove duplicates, consistent casing)
        if 'skills' in data and isinstance(data['skills'], list):
            normalized_skills = []
            seen_skills = set()
            for skill in data['skills']:
                skill_lower = skill.lower().strip()
                if skill_lower not in seen_skills:
                    seen_skills.add(skill_lower)
                    normalized_skills.append(skill.strip().title())
            data['skills'] = sorted(normalized_skills)  # Sort for consistency
        
        # Normalize industry
        if 'industry' in data:
            data['industry'] = data['industry'].strip().title()
        
        return data

# Enhanced ATS Scoring Agent with deterministic scoring
class ATSScoringAgent:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.agent = autogen.AssistantAgent(
            name="ATSScorer",
            system_message="""You are an ATS scoring specialist. Provide ABSOLUTELY CONSISTENT, numerical scores.

CRITICAL: You MUST be deterministic - same input = identical output EVERY TIME.

Scoring weights (MUST follow exactly):
- Skills Match: 30% (Count exact matches, partial matches get 0.5 weight)
- Experience Relevance: 25% (Years + domain relevance)
- Education Alignment: 15% (Degree level + field relevance)
- Format and Structure: 15% (Consistent baseline score)
- Keyword Optimization: 15% (Keyword density calculation)

SCORING ALGORITHM - BE MATHEMATICAL AND CONSISTENT:
1. Skills: (exact_matches + 0.5*partial_matches) / total_required_skills * 100
2. Experience: min(years_experience/required_years, 1.0) * domain_match_score * 100
3. Education: degree_level_match * field_relevance_score * 100
4. Format: 75 (baseline for structured resume)
5. Keywords: keyword_count / total_possible_keywords * 100

CRITICAL: Return ONLY valid JSON in this EXACT format - no additional text:
{
    "total_score": 75.5,
    "skills_score": 80.0,
    "experience_score": 70.0,
    "education_score": 75.0,
    "format_score": 85.0,
    "keyword_score": 65.0,
    "explanation": "Brief scoring rationale explaining the mathematical calculation",
    "benchmark_comparison": "Above average for software engineering roles",
    "confidence_level": 0.95
}

BE DETERMINISTIC: Identical inputs must produce identical scores. Use mathematical formulas, not subjective judgment.
""",
            llm_config={"config_list": llm_config, "timeout": 10, "temperature": 0.0}  # Zero temperature
        )
    
    def calculate_score(self, resume_data: Dict, job_requirements: Dict, content_hash: str = "", job_hash: str = "") -> Dict[str, Any]:
        """Calculate ATS score with caching for consistency"""
        # Check if we already have a score for this exact combination
        if content_hash and job_hash:
            cached_score = self.db.check_existing_score(content_hash, job_hash) if hasattr(self, 'db') else None
            if cached_score:
                st.info("🎯 Retrieved consistent score from cache")
                return cached_score
        
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        
        # Create deterministic input by sorting
        sorted_resume = json.dumps(resume_data, sort_keys=True)
        sorted_job = json.dumps(job_requirements, sort_keys=True)
        
        message = f"""
Score this resume against job requirements using MATHEMATICAL calculations for consistency.
Use zero temperature and be absolutely deterministic. Return only JSON.

RESUME DATA (sorted for consistency):
{sorted_resume[:2000]}

JOB REQUIREMENTS (sorted for consistency):
{sorted_job[:1500]}

Apply the mathematical scoring formulas exactly as specified.
"""
        
        try:
            start_time = time.time()
            chat_result = user_proxy.initiate_chat(self.agent, message=message, max_turns=1)
            processing_time = time.time() - start_time
            
            response = chat_result.chat_history[-1]["content"]
            
            extracted_data = extract_json_from_response(response)
            if extracted_data and 'total_score' in extracted_data:
                extracted_data['processing_time'] = processing_time
                return extracted_data
                
        except Exception as e:
            st.error(f"Scoring error: {e}")
        
        # Deterministic fallback scoring based on simple rules
        return self._fallback_deterministic_scoring(resume_data, job_requirements)
    
    def _fallback_deterministic_scoring(self, resume_data: Dict, job_requirements: Dict) -> Dict[str, Any]:
        """Deterministic fallback scoring using mathematical rules"""
        # Skills scoring
        resume_skills = set(skill.lower() for skill in resume_data.get('skills', []))
        required_skills = set(skill.lower() for skill in job_requirements.get('required_skills', []))
        
        if required_skills:
            skill_matches = len(resume_skills.intersection(required_skills))
            skills_score = min(100, (skill_matches / len(required_skills)) * 100)
        else:
            skills_score = 75.0
        
        # Experience scoring (simplified)
        experience_score = 70.0  # Default based on having experience entries
        if resume_data.get('experience'):
            experience_score = 75.0
        
        # Education scoring
        education_score = 65.0
        if resume_data.get('education'):
            education_score = 70.0
        
        # Format scoring (consistent baseline)
        format_score = 75.0
        
        # Keyword scoring
        all_text = ' '.join([str(v) for v in resume_data.values() if isinstance(v, (str, list))])
        keywords = job_requirements.get('key_keywords', [])
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw.lower() in all_text.lower())
            keyword_score = min(100, (keyword_matches / len(keywords)) * 100)
        else:
            keyword_score = 60.0
        
        # Calculate weighted total
        total_score = (
            skills_score * 0.3 +
            experience_score * 0.25 +
            education_score * 0.15 +
            format_score * 0.15 +
            keyword_score * 0.15
        )
        
        return {
            "total_score": round(total_score, 1),
            "skills_score": round(skills_score, 1),
            "experience_score": round(experience_score, 1),
            "education_score": round(education_score, 1),
            "format_score": round(format_score, 1),
            "keyword_score": round(keyword_score, 1),
            "explanation": "Mathematical fallback scoring applied for consistency",
            "benchmark_comparison": "Average for general roles",
            "confidence_level": 0.95,
            "processing_time": 0
        }

# Enhanced Job Description Analysis Agent with consistent parsing
class JobAnalysisAgent:
    def __init__(self, llm_config, db: ResumeDatabase):
        self.llm_config = llm_config
        self.db = db
        self.agent = autogen.AssistantAgent(
            name="JobAnalyzer",
            system_message="""You are a job description analysis expert. Parse job descriptions with ABSOLUTE CONSISTENCY.

CRITICAL: Same job description must produce identical results every time.

Your tasks:
1. Extract required and preferred skills (normalize and deduplicate)
2. Identify experience requirements (years, level, domains)
3. Determine education requirements
4. Find key keywords for ATS matching (comprehensive list)
5. Identify industry and role type (standardized categories)

CONSISTENCY RULES:
- Always extract skills in alphabetical order
- Standardize skill names (e.g., "Javascript" -> "JavaScript")
- Use consistent industry categories
- Extract ALL relevant keywords, not just a subset

CRITICAL: Return ONLY valid JSON in this EXACT format - no additional text:
{
    "required_skills": ["Python", "JavaScript", "SQL"],
    "preferred_skills": ["React", "Node.js"],
    "experience_requirements": {
        "years": 3,
        "level": "mid",
        "domains": ["web development", "software engineering"]
    },
    "education_requirements": {
        "degree_level": "bachelor",
        "fields": ["Computer Science", "Engineering"]
    },
    "key_keywords": ["software", "development", "programming", "coding", "algorithms"],
    "industry": "Technology",
    "role_type": "Software Engineer"
}

Be thorough and consistent in keyword extraction. Same job description = identical output.
""",
            llm_config={"config_list": llm_config, "timeout": 30, "temperature": 0.0}  # Zero temperature
        )
    
    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description with consistency and caching"""
        # Normalize job description
        normalized_job = normalize_text(job_description)
        job_hash = self.db.get_content_hash(job_description)
        
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        
        message = f"""
Analyze this job description and extract structured requirements with absolute consistency.
Return only the JSON structure with no additional text.

JOB DESCRIPTION (normalized):
{normalized_job[:2500]}
"""
        
        try:
            chat_result = user_proxy.initiate_chat(self.agent, message=message, max_turns=1)
            response = chat_result.chat_history[-1]["content"]
            
            extracted_data = extract_json_from_response(response)
            if extracted_data:
                # Normalize extracted data
                extracted_data = self._normalize_job_data(extracted_data)
                return extracted_data
                
        except Exception as e:
            st.error(f"Job analysis error: {e}")
        
        # Fallback structure
        return {
            "required_skills": [],
            "preferred_skills": [],
            "experience_requirements": {"years": 0, "level": "entry", "domains": []},
            "education_requirements": {"degree_level": "bachelor", "fields": []},
            "key_keywords": [],
            "industry": "General",
            "role_type": "General"
        }
    
    def _normalize_job_data(self, data: Dict) -> Dict:
        """Normalize job analysis data for consistency"""
        # Normalize skills lists
        for skill_key in ['required_skills', 'preferred_skills']:
            if skill_key in data and isinstance(data[skill_key], list):
                normalized_skills = [skill.strip().title() for skill in data[skill_key]]
                data[skill_key] = sorted(list(set(normalized_skills)))  # Remove duplicates and sort
        
        # Normalize keywords
        if 'key_keywords' in data and isinstance(data['key_keywords'], list):
            normalized_keywords = [kw.strip().lower() for kw in data['key_keywords']]
            data['key_keywords'] = sorted(list(set(normalized_keywords)))
        
        return data

# Enhanced Improvement Recommendation Agent
class ImprovementAgent:
    def __init__(self, llm_config, db: ResumeDatabase):
        self.llm_config = llm_config
        self.db = db
        self.agent = autogen.AssistantAgent(
            name="ImprovementAdvisor",
            system_message="""You are a resume improvement specialist. Provide specific, actionable recommendations with consistency.

Your tasks:
1. Identify missing keywords that would improve ATS scores
2. Find skill gaps between resume and job requirements
3. Suggest format and structure improvements
4. Recommend content enhancements
5. Prioritize the most impactful improvements
6. Provide industry-specific advice

CRITICAL: Return ONLY valid JSON in this EXACT format - no additional text:
{
    "missing_keywords": ["keyword1", "keyword2", "keyword3"],
    "skill_gaps": ["skill1", "skill2"],
    "format_improvements": ["improvement1", "improvement2"],
    "content_enhancements": ["enhancement1", "enhancement2"],
    "priority_actions": ["Most important action", "Second priority", "Third priority"],
    "industry_specific_advice": ["advice1", "advice2"]
}

Be specific and actionable. Focus on improvements that will increase ATS scores.
""",
            llm_config={"config_list": llm_config, "timeout": 30, "temperature": 0.1}
        )
    
    def generate_recommendations(self, resume_data: Dict, job_requirements: Dict, scores: Dict) -> Dict[str, Any]:
        """Generate improvement recommendations with RAG-enhanced industry insights"""
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        
        # Get industry-specific best practices from RAG
        industry = resume_data.get('industry', 'General')
        best_practices = self.db.get_industry_best_practices(industry)
        
        message = f"""
Based on this data, provide specific improvement recommendations.
Focus on actionable steps to improve ATS scores. Return only JSON with no additional text.

RESUME DATA:
{json.dumps(resume_data, indent=2)[:1500]}

JOB REQUIREMENTS:
{json.dumps(job_requirements, indent=2)[:1500]}

CURRENT SCORES:
{json.dumps(scores, indent=2)}

INDUSTRY BEST PRACTICES:
{json.dumps(best_practices, indent=2)[:1000]}
"""
        
        try:
            chat_result = user_proxy.initiate_chat(self.agent, message=message, max_turns=1)
            response = chat_result.chat_history[-1]["content"]
            
            extracted_data = extract_json_from_response(response)
            if extracted_data:
                return extracted_data
                
        except Exception as e:
            st.error(f"Recommendation error: {e}")
        
        # Fallback recommendations with gap analysis
        return self._generate_fallback_recommendations(resume_data, job_requirements, scores)
    
    def _generate_fallback_recommendations(self, resume_data: Dict, job_requirements: Dict, scores: Dict) -> Dict:
        """Generate fallback recommendations based on gap analysis"""
        resume_skills = set(skill.lower() for skill in resume_data.get('skills', []))
        required_skills = set(skill.lower() for skill in job_requirements.get('required_skills', []))
        
        # Find missing skills
        missing_skills = list(required_skills - resume_skills)
        
        # Find missing keywords
        resume_text = json.dumps(resume_data).lower()
        job_keywords = job_requirements.get('key_keywords', [])
        missing_keywords = [kw for kw in job_keywords if kw.lower() not in resume_text]
        
        return {
            "missing_keywords": missing_keywords[:5],
            "skill_gaps": missing_skills[:5],
            "format_improvements": ["Add quantified achievements", "Include relevant certifications"],
            "content_enhancements": ["Expand technical project descriptions", "Add industry-specific terminology"],
            "priority_actions": [
                f"Add missing skills: {', '.join(missing_skills[:3])}",
                f"Include keywords: {', '.join(missing_keywords[:3])}",
                "Quantify achievements with metrics"
            ],
            "industry_specific_advice": ["Follow industry resume formatting standards", "Include relevant certifications"]
        }

# Enhanced Visualization Agent with improved heatmap
class VisualizationAgent:
    def __init__(self):
        pass
    
    def create_score_comparison_chart(self, scores: Dict) -> go.Figure:
        """Create professional score comparison visualization"""
        categories = ['Skills Match', 'Experience', 'Education', 'Format', 'Keywords']
        values = [
            scores.get('skills_score', 0),
            scores.get('experience_score', 0),
            scores.get('education_score', 0),
            scores.get('format_score', 0),
            scores.get('keyword_score', 0)
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Score',
            fillcolor='rgba(75, 192, 192, 0.2)',
            line=dict(color='rgba(75, 192, 192, 1)')
        ))
        
        # Add industry average (benchmark)
        benchmark_values = [70, 65, 60, 75, 68]  # Typical industry averages
        fig.add_trace(go.Scatterpolar(
            r=benchmark_values,
            theta=categories,
            fill='toself',
            name='Industry Average',
            fillcolor='rgba(255, 99, 132, 0.2)',
            line=dict(color='rgba(255, 99, 132, 1)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="ATS Score Comparison vs Industry Average"
        )
        
        return fig

    def create_skill_match_heatmap(self, resume_skills: List[str], required_skills: List[str]) -> go.Figure:
        """Create an enhanced skill matching heatmap with advanced matching"""
        
        # Ensure we have skills to work with
        if not resume_skills:
            resume_skills = ["No skills found"]
        if not required_skills:
            required_skills = ["No requirements found"]
        
        # Limit to top 10 for readability but ensure we show something
        display_required = required_skills[:10] if len(required_skills) >= 10 else required_skills
        display_resume = resume_skills[:10] if len(resume_skills) >= 10 else resume_skills
        
        # Use advanced matching algorithm
        match_matrix = advanced_skill_matching(display_resume, display_required)
        
        # If still no valid matrix, create a basic one
        if match_matrix.size == 0:
            match_matrix = np.random.rand(len(display_required), len(display_resume)) * 0.3  # Low random values
        
        # Ensure proper dimensions
        if match_matrix.ndim == 1:
            match_matrix = match_matrix.reshape(-1, 1)
        
        # Create the heatmap with better color scale
        fig = go.Figure(data=go.Heatmap(
            z=match_matrix,
            x=display_resume,
            y=display_required,
            colorscale=[
                [0, 'rgb(255,255,255)'],      # White for no match
                [0.3, 'rgb(255,200,200)'],    # Light red for weak match
                [0.6, 'rgb(255,165,0)'],      # Orange for moderate match
                [0.8, 'rgb(144,238,144)'],    # Light green for good match
                [1, 'rgb(0,128,0)']           # Dark green for perfect match
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Match Strength",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['No Match', 'Weak', 'Moderate', 'Good', 'Perfect']
            )
        ))
        
        fig.update_layout(
            title='Resume vs. Job Description Skill Match Analysis',
            xaxis_title="Resume Skills",
            yaxis_title="Required Skills",
            height=400,
            margin=dict(l=150, r=50, t=80, b=100)
        )
        
        # Add annotations for high matches
        annotations = []
        for i in range(match_matrix.shape[0]):
            for j in range(match_matrix.shape[1]):
                if match_matrix[i, j] > 0.7:  # Only annotate good matches
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=f"{match_matrix[i, j]:.2f}",
                            showarrow=False,
                            font=dict(color="white", size=12)
                        )
                    )
        
        fig.update_layout(annotations=annotations)
        return fig
    
    def create_improvement_priority_chart(self, recommendations: Dict) -> go.Figure:
        """Create a priority chart for improvements"""
        priorities = recommendations.get('priority_actions', [])
        if not priorities:
            priorities = ['No specific priorities identified']
        
        # Create impact scores (mock data for visualization)
        impact_scores = [90, 75, 60, 45, 30][:len(priorities)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=impact_scores,
                y=priorities,
                orientation='h',
                marker=dict(
                    color=impact_scores,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Impact Score")
                )
            )
        ])
        
        fig.update_layout(
            title='Improvement Priority by Impact',
            xaxis_title='Estimated Impact Score',
            yaxis_title='Improvement Actions',
            height=300
        )
        
        return fig

# Enhanced Main ATS System Class
class ProfessionalATSSystem:
    def __init__(self, llm_config_list):
        self.llm_config = llm_config_list
        self.db = ResumeDatabase()
        
        # Test connection first
        connection_ok, message = test_llm_connection(llm_config_list)
        if not connection_ok:
            st.error(f"LLM connection failed: {message}")
            self.initialized = False
            return
        
        # Initialize all agents with database reference for consistency
        self.resume_processor = ResumeProcessingAgent(llm_config_list, self.db)
        self.ats_scorer = ATSScoringAgent(llm_config_list)
        self.ats_scorer.db = self.db  # Add database reference for caching
        self.job_analyzer = JobAnalysisAgent(llm_config_list, self.db)
        self.improvement_agent = ImprovementAgent(llm_config_list, self.db)
        self.visualization_agent = VisualizationAgent()
        self.initialized = True
        
        st.success("✅ All 5 AutoGen agents initialized successfully!")
    
    def full_resume_analysis(self, resume_text: str, job_description: str, filename: str = "uploaded_resume", file_size: int = 0) -> Dict[str, Any]:
        """Complete resume analysis workflow"""
        if not self.initialized:
            return {}
        
        try:
            start_time = time.time()
            
            # Generate consistency hashes
            content_hash = self.db.get_content_hash(resume_text)
            job_hash = self.db.get_content_hash(job_description)
            
            st.info("📝 Step 1/4: Processing resume with AutoGen...")
            process_start = time.time()
            resume_data = self.resume_processor.process_resume(resume_text)
            process_time = time.time() - process_start
            
            if process_time > 30:
                st.warning(f"⚠️ Resume processing took {process_time:.1f}s (target: <30s)")
            
            st.info("📋 Step 2/4: Analyzing job description...")
            job_requirements = self.job_analyzer.analyze_job_description(job_description)
            
            st.info("📊 Step 3/4: Calculating ATS scores...")
            score_start = time.time()
            scores = self.ats_scorer.calculate_score(resume_data, job_requirements, content_hash, job_hash)
            score_time = time.time() - score_start
            
            if score_time > 10:
                st.warning(f"⚠️ Score calculation took {score_time:.1f}s (target: <10s)")
            
            st.info("💡 Step 4/4: Generating improvement recommendations...")
            recommendations = self.improvement_agent.generate_recommendations(
                resume_data, job_requirements, scores
            )
            
            total_time = time.time() - start_time
            
            # Save to database for consistency tracking
            resume_id = self.db.save_resume(filename, resume_text, resume_data, file_size)
            self.db.save_scores(resume_id, scores, content_hash, job_hash, total_time)
            
            # Get job recommendations using enhanced RAG
            resume_skills = resume_data.get('skills', [])
            recommended_jobs = self.db.get_relevant_jobs(resume_skills)
            
            st.success(f"🎉 Analysis complete!")
            
            return {
                "resume_data": resume_data,
                "job_requirements": job_requirements,
                "scores": scores,
                "recommendations": recommendations,
                "resume_id": resume_id,
                "content_hash": content_hash,
                "job_hash": job_hash,
               # "processing_metrics": {
                #    "total_time": total_time,
                 #   "process_time": process_time,
                  #  "score_time": score_time
                 "recommended_jobs": recommended_jobs
            }
            
        except Exception as e:
            st.error(f"Analysis workflow error: {e}")
            return {}

# Export Functions
def create_pdf_report(results: Dict, filename: str) -> bytes:
    """Create comprehensive PDF report and return as bytes object."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    title = Paragraph(f"Professional ATS Resume Analysis Report", styles['Title'])
    story.append(title)
    
    subtitle = Paragraph(f"Generated for: {filename}", styles['Heading2'])
    story.append(subtitle)
    story.append(Spacer(1, 0.2 * inch))

    scores = results.get('scores', {})
    recommendations = results.get('recommendations', {})
    
    story.append(Paragraph(f"<b>Overall ATS Score:</b> {scores.get('total_score', 0):.1f}/100", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("<b>Category Breakdown:</b>", styles['Normal']))
    story.append(Paragraph(f"• Skills Match (30%): {scores.get('skills_score', 0):.1f}/100", styles['Normal']))
    story.append(Paragraph(f"• Experience (25%): {scores.get('experience_score', 0):.1f}/100", styles['Normal']))
    story.append(Paragraph(f"• Education (15%): {scores.get('education_score', 0):.1f}/100", styles['Normal']))
    story.append(Paragraph(f"• Format (15%): {scores.get('format_score', 0):.1f}/100", styles['Normal']))
    story.append(Paragraph(f"• Keywords (15%): {scores.get('keyword_score', 0):.1f}/100", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def create_comprehensive_csv(results: Dict) -> str:
    """Create a comprehensive CSV string from analysis results."""
    data = []
    headers = ['Category', 'Metric', 'Value', 'Context']
    
    scores = results.get('scores', {})
    for category, score in scores.items():
        data.append({
            'Category': 'Scores',
            'Metric': category.replace('_', ' ').title(),
            'Value': score,
            'Context': scores.get('explanation', '') if category == 'total_score' else ''
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# Cache the ATS system instance
@st.cache_resource
def get_professional_ats_system(provider: str, api_key: str):
    """Initialize and cache the professional ATS system"""
    config_list = get_free_llm_config(provider, api_key)
    if not config_list:
        return None
    return ProfessionalATSSystem(config_list)

# Enhanced UI functions
def display_executive_summary(results: Dict):
    """Executive summary display"""
    st.markdown("### 📄 Executive Summary")

    scores = results.get('scores', {})
    total_score = scores.get('total_score', 0)
    
    st.markdown(f"#### Overall Match Score: **{total_score:.1f}/100**")
    
    if total_score >= 80:
        st.success("This resume is an **excellent match** for the job requirements.")
    elif total_score >= 60:
        st.warning("This resume is a **good match**, but has room for improvement.")
    else:
        st.error("This resume is a **poor match**. Significant improvements needed.")
    
    st.markdown("---")
    
    st.markdown("#### 🌟 Key Highlights")
    kpi_cols = st.columns(3)
    
    with kpi_cols[0]:
        st.metric("Skills Match", f"{scores.get('skills_score', 0):.1f}%")
    with kpi_cols[1]:
        st.metric("Experience Relevance", f"{scores.get('experience_score', 0):.1f}%")
    with kpi_cols[2]:
        st.metric("Keyword Optimization", f"{scores.get('keyword_score', 0):.1f}%")
    
# Main Application
def main():
    st.set_page_config(
        page_title="Professional ATS Scoring System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown("""
        <div class="main-header">
            <h1>🎯 Professional ATS Resume Scoring System</h1>
            <p>Comprehensive AutoGen-based ATS with 5 Specialized Agents</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("🤖 LLM Provider Configuration")
    provider = st.sidebar.selectbox(
        "Select Free LLM Provider",
        options=["groq", "deepseek", "together", "local_ollama"],
        help="Choose your preferred free LLM provider"
    )

    if provider == "groq":
        st.sidebar.info("🚀 **Groq** - Fastest option")
        api_key = st.sidebar.text_input("Groq API Key", type="password")
    elif provider == "deepseek":
        st.sidebar.info("🧠 **DeepSeek** - High-quality analysis")
        api_key = st.sidebar.text_input("DeepSeek API Key", type="password")
    elif provider == "together":
        st.sidebar.info("🤝 **Together AI** - Reliable processing")
        api_key = st.sidebar.text_input("Together API Key", type="password")
    else:
        st.sidebar.info("💻 **Ollama** - Local processing")
        api_key = "ollama"

    # Initialize system
    ats_system = None
    if api_key:
        ats_system = get_professional_ats_system(provider, api_key)
        if ats_system and ats_system.initialized:
            st.sidebar.success("✅ LLM system initialized successfully!")
        elif ats_system and not ats_system.initialized:
            st.sidebar.error("❌ System initialization failed")
            ats_system = None
    else:
        st.sidebar.warning("⚠️ Please enter API key")

    # Main content
    st.header("📝 Upload & Analyze")
    
    # File upload and text input
    col_file, col_job = st.columns(2)
    
    with col_file:
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"],
            help="Upload your resume for analysis"
        )
        
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name} uploaded ({uploaded_file.size} bytes)")
    
    with col_job:
        job_description = st.text_area(
            "Job Description", 
            height=200,
            help="Paste the complete job description here",
            placeholder="Paste job description here..."
        )

    # Analysis button
    run_analysis = st.button("🚀 Run ATS Analysis", use_container_width=True, disabled=not ats_system)

    # Main Analysis
    results = {}
    if run_analysis and ats_system and (uploaded_file or job_description):
        if not job_description:
            st.error("Please provide a job description")
            return
            
        with st.spinner("📄 Running ATS analysis..."):
            resume_text = ""
            filename = "manual_input"
            file_size = 0
            
            if uploaded_file:
                resume_text = parse_resume_text_from_upload(
                    uploaded_file, uploaded_file.name.split('.')[-1]
                )
                filename = uploaded_file.name
                file_size = uploaded_file.size
                
                if not resume_text:
                    st.error("Failed to extract text from file")
                    return
            
            results = ats_system.full_resume_analysis(resume_text, job_description, filename, file_size)
            
            if results and results.get('scores'):
                st.session_state['results'] = results
                st.session_state['filename'] = filename
                st.session_state['resume_text'] = resume_text
                st.session_state['job_description'] = job_description
            else:
                st.error("❌ Analysis failed")

    # Display results
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        filename = st.session_state['filename']
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Executive Summary", 
            "📊 Score Breakdown", 
            "💡 Improvements", 
            "📋 Job Analysis",
            "📤 Export"
        ])
        
        with tab1:
            display_executive_summary(results)
            
            # Add job recommendations
            st.markdown("---")
            st.markdown("#### 🎯 Recommended Jobs")
            recommended_jobs = results.get('recommended_jobs', [])
            if recommended_jobs:
                for i, job in enumerate(recommended_jobs[:3]):
                    with st.expander(f"🏢 {job['title']} at {job['company']} - {job['match_score']:.1f}% match"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Industry:** {job['industry']}")
                            st.write(f"**Location:** {job['location']}")
                        with col2:
                            st.write(f"**Salary:** {job['salary_range']}")
                            st.write(f"**Match Score:** {job['match_score']:.1f}%")
                        st.write(f"**Requirements:** {job['requirements']}")
            else:
                st.info("No job recommendations available")
        
        with tab2:
            st.markdown("### 📊 Score Breakdown")
            
            scores = results.get('scores', {})
            
            st.markdown(f"#### Total Score: **{scores.get('total_score', 0):.1f}/100**")
            
            st.info(scores.get('explanation', 'No detailed explanation provided'))
            
            # Enhanced visualizations
            viz_cols = st.columns(2)
            with viz_cols[0]:
                score_chart = ats_system.visualization_agent.create_score_comparison_chart(scores)
                st.plotly_chart(score_chart, use_container_width=True)
                
            with viz_cols[1]:
                job_reqs = results.get('job_requirements', {})
                resume_data = results.get('resume_data', {})
                skill_heatmap = ats_system.visualization_agent.create_skill_match_heatmap(
                    resume_data.get('skills', []),
                    job_reqs.get('required_skills', [])
                )
                st.plotly_chart(skill_heatmap, use_container_width=True)
            
            # Detailed metrics
            st.markdown("#### 🔍 Detailed Breakdown")
            
            metric_cols = st.columns(5)
            metrics_data = [
                ("Skills Match", scores.get('skills_score', 0), "30% weight"),
                ("Experience", scores.get('experience_score', 0), "25% weight"),
                ("Education", scores.get('education_score', 0), "15% weight"),
                ("Format", scores.get('format_score', 0), "15% weight"),
                ("Keywords", scores.get('keyword_score', 0), "15% weight")
            ]
            
            for i, (label, value, weight) in enumerate(metrics_data):
                with metric_cols[i]:
                    st.metric(label, f"{value:.1f}%", help=weight)
        
        with tab3:
            st.markdown("### 💡 Personalized Improvement Plan")
            
            recommendations = results.get('recommendations', {})
            
            # Priority actions with impact visualization
            st.markdown("#### 🎯 Priority Actions")
            if recommendations.get('priority_actions'):
                priority_chart = ats_system.visualization_agent.create_improvement_priority_chart(recommendations)
                st.plotly_chart(priority_chart, use_container_width=True)
                
                for i, action in enumerate(recommendations['priority_actions'], 1):
                    st.info(f"**{i}.** {action}")
            
            # Detailed recommendations
            rec_cols = st.columns(2)
            
            with rec_cols[0]:
                st.markdown("##### 🔍 Missing Keywords")
                missing_kw = recommendations.get('missing_keywords', [])
                if missing_kw:
                    for kw in missing_kw:
                        st.warning(f"• **{kw}**")
                else:
                    st.success("✅ No critical keywords missing")
                
                st.markdown("##### 📚 Skill Gaps")
                skill_gaps = recommendations.get('skill_gaps', [])
                if skill_gaps:
                    for skill in skill_gaps:
                        st.error(f"• **{skill}**")
                else:
                    st.success("✅ No major skill gaps identified")
            
            with rec_cols[1]:
                st.markdown("##### 📝 Content Enhancements")
                enhancements = recommendations.get('content_enhancements', [])
                for enhancement in enhancements:
                    st.info(f"• {enhancement}")
                
                st.markdown("##### 🏭 Industry-Specific Advice")
                industry_advice = recommendations.get('industry_specific_advice', [])
                for advice in industry_advice:
                    st.info(f"• {advice}")
        
        with tab4:
            st.markdown("### 📋 Job Requirements Analysis")
            
            job_reqs = results.get('job_requirements', {})
            
            # Requirements overview
            req_cols = st.columns(2)
            
            with req_cols[0]:
                st.markdown("##### 🎯 Required Skills")
                required_skills = job_reqs.get('required_skills', [])
                if required_skills:
                    for skill in required_skills:
                        st.markdown(f"• {skill}")
                else:
                    st.info("No specific required skills identified")
                
                st.markdown("##### 💼 Experience Requirements")
                exp_req = job_reqs.get('experience_requirements', {})
                st.write(f"**Years:** {exp_req.get('years', 'Not specified')}")
                st.write(f"**Level:** {exp_req.get('level', 'Not specified').title()}")
                domains = exp_req.get('domains', [])
                if domains:
                    st.write(f"**Domains:** {', '.join(domains)}")
            
            with req_cols[1]:
                st.markdown("##### ⭐ Preferred Skills")
                preferred_skills = job_reqs.get('preferred_skills', [])
                if preferred_skills:
                    for skill in preferred_skills:
                        st.markdown(f"• {skill}")
                else:
                    st.info("No preferred skills specified")
                
                st.markdown("##### 🎓 Education Requirements")
                edu_req = job_reqs.get('education_requirements', {})
                st.write(f"**Degree Level:** {edu_req.get('degree_level', 'Not specified').title()}")
                fields = edu_req.get('fields', [])
                if fields:
                    st.write(f"**Fields:** {', '.join(fields)}")
            
            # Keywords analysis
            st.markdown("##### 🔑 Key ATS Keywords")
            keywords = job_reqs.get('key_keywords', [])
            if keywords:
                # Display keywords in a nice format
                keyword_text = " • ".join(keywords)
                st.markdown(f"```\n{keyword_text}\n```")
            else:
                st.info("No specific keywords identified")
        
        with tab5:
            st.markdown("### 📤 Export & Share Results")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                st.markdown("**📄 PDF Report**")
                if st.button("Generate PDF", use_container_width=True):
                    pdf_content = create_pdf_report(results, filename)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_content,
                        file_name=f"ats_analysis_{filename.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            with export_cols[1]:
                st.markdown("**📊 CSV Data**")
                if st.button("Generate CSV", use_container_width=True):
                    csv_content = create_comprehensive_csv(results)
                    st.download_button(
                        label="⬇️ Download CSV Data",
                        data=csv_content,
                        file_name=f"ats_data_{filename.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_cols[2]:
                st.markdown("**📋 JSON Data**")
                if st.button("Generate JSON", use_container_width=True):
                    json_content = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="⬇️ Download JSON Data",
                        data=json_content,
                        file_name=f"ats_results_{filename.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # Export preview
            st.markdown("---")
            st.markdown("**📋 Quick Export Preview**")
            
            preview_data = {
                'Total Score': results.get('scores', {}).get('total_score', 0),
                'Skills Score': results.get('scores', {}).get('skills_score', 0),
                'Experience Score': results.get('scores', {}).get('experience_score', 0),
                'Analysis Time': f"{results.get('processing_metrics', {}).get('total_time', 0):.2f}s"
            }
            
            preview_df = pd.DataFrame(list(preview_data.items()), columns=['Metric', 'Value'])
            st.dataframe(preview_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Powered by AutoGen & Streamlit"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()