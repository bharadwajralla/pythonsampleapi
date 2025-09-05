import base64
import glob
from pathlib import Path
import re
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
from sonarqube import SonarQubeClient
from abc import ABC, abstractmethod
import git
import tempfile
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM

# ==============================
# Hugging Face LLM Helper
# =========================

HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN")
GITHUB_TOKEN=os.getenv("GITHUB_API")
SONAR_TOKEN=os.getenv("SONAR_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/codet5p-220m"
HF_HEADERS = {"Authorization":f"Bearer {HUGGING_FACE_TOKEN}","Content-Type": "application/json"}  # Replace with HF token
SONAR_URL=os.getenv("SONAR_API")

MODEL_ID = "Salesforce/codet5p-220m"
    # Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)  

async def hf_generate(prompt, max_tokens=400):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# Agent Base Class
# ==============================
class Agent(ABC):
    @abstractmethod
    def run(self, input_data):
        pass

#================================
#Sensetive AI Pattern
#================================
class SensitivePatternAI:
    def __init__(self, knowledge_base_file=None):
        # Use HuggingFace zero-shot model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Load seed terms from knowledge base file or AI generation
        if knowledge_base_file and knowledge_base_file.endswith(".txt"):
            with open(knowledge_base_file, "r", encoding="utf-8") as f:
                base_terms = [line.strip() for line in f if line.strip()]
        else:
            # Ask AI to generate sensitive programming keywords dynamically
            base_terms = self.generate_keywords_with_ai()

        # Expand into regex patterns
        self.patterns = self.generate_patterns(base_terms)

    def generate_keywords_with_ai(self):
        """Use AI to propose sensitive programming keywords."""
        candidate_labels = ["security", "credentials", "secret", "private", "authentication"]
        text = "List sensitive terms in programming such as passwords, tokens, or keys"
        result = self.classifier(text, candidate_labels)

        # Simulate extracted terms from model (mock for free HuggingFace pipeline)
        terms = ["password", "apikey", "secret", "token", "certificate", "private_key", "sessionid"]
        return terms

    def generate_patterns(self, terms):
        """Turn terms into regex patterns dynamically."""
        patterns = []
        for word in terms:
            regex_variants = [
                rf"(?i){word}\s*=\s*['\"].+['\"]",   # assignment
                rf"(?i){word}[_\-]?[A-Za-z0-9]*",    # extended names
                rf"(?i){word}"                       # plain usage
            ]
            patterns.extend(regex_variants)
        return list(set(patterns))

    def contains_sensitive_data(self, text):
        return any(re.search(pattern, text) for pattern in self.patterns)

# ==============================
# Agents
# ==============================
class DispatcherAgent(Agent):
    def run(self, input_data):
        prompt = f"""Classify this developer request into one of:
        - code_suggestion
        - code_quality
        Request: {input_data}
        Answer with only the label."""
        return hf_generate(prompt, 50).strip().lower()
    
    def clone_repo(self, url: str) -> str:
        tmpdir = tempfile.mkdtemp()
        git.Repo.clone_from(url, tmpdir)
        return tmpdir
    
    def find_config_file(self, repo_path: str) -> list:
        PROGRAMING_PATTERNS={
            ".Net":["web.config,app.config,appsettings.json"],
            "Avm":[".cfg.json", ".config",".content.xml", ".xml", ".json"]
        }
        
        
        for root, _, files in os.walk(repo_path):
            for f in files:
               file_path=Path(root)/f
               file_lower = f.lower()
            content = None
            file_format = file_path.suffix.lower()
            for cfg_type, extensions in PROGRAMING_PATTERNS.items():
                   if(cfg_type==".Net"):
                       if file_lower in extensions:
                             with open(f"{root}/{f}", "r", encoding="utf-8") as file:
                                content=file.read()
                                return content
                   elif file_format in extensions:
                    with open(f"{root}/{f}", "r", encoding="utf-8") as file:
                       content=file.read()
                       return content
                    break
        return content

    def mask_sensitive(self, content: str) -> str:
        patterns = [
            r"([A-Za-z0-9]{20,})",
            r"(password\s*=\s*['\"].*?['\"])",
            r"(api[_-]?key\s*=\s*['\"].*?['\"])",
            r"(secret\s*=\s*['\"].*?['\"])",
        ]
        masked = content
        for p in patterns:
            masked = re.sub(p, "****", masked, flags=re.IGNORECASE)
        return masked


    async def send_to_hf(self, masked_config: str, prompt: str) -> Dict:
        headerprompt=f"""
            You are an AI code assistant and give the resposne as json.
            Detect progaming language and version from this below sample config /n #Sample Config /n'{masked_config}'

            Output format: 
            {{"language":"language","version":"version"}}
            """
        response = await hf_generate(headerprompt,400)
        return response

    def extract_best_practices(self, lang: str, version: str, prompt: str, top_k=3) -> List[str]:
        repo_path = os.path.join(self.base_path, lang, version)

        if not os.path.exists(repo_path):
            repo_path = os.path.join(self.base_path, lang, "default")

        docs = []
        for file in glob.glob(os.path.join(repo_path, "**/*.txt"), recursive=True):
            with open(file, "r", encoding="utf-8") as f:
                docs.extend([p.strip() for p in f.read().split("\n\n") if p.strip()])

        if not docs:
            return []

        vectorizer = TfidfVectorizer().fit_transform(docs + [prompt])
        vectors = vectorizer.toarray()
        query_vec = vectors[-1].reshape(1, -1)

        sims = cosine_similarity(query_vec, vectors[:-1])[0]
        ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


class CodeSuggestionAgent(Agent):
    def run(self, input_data,lang):
        prompt = f"Generate 2 different {lang} code solutions for this request:\n{input_data}"
        raw = hf_generate(prompt, 400)
        return [s.strip() for s in raw.split("```") if s.strip()]


class RankingAgent(Agent):
    def run(self, suggestions):
        joined = "\n\n".join([f"Solution {i+1}:\n{s}" for i, s in enumerate(suggestions)])
        prompt = f"""Rank these code solutions by correctness, readability, and efficiency.
        Put the best solution first:\n{joined}"""
        ranked = hf_generate(prompt, 300)
        return ranked.split("\n\n")


class CodeQualityAgent(Agent):
    def __init__(self, sonar_url=F"{SONAR_URL}", sonar_token=f"{SONAR_TOKEN}"):
        self.sonar = SonarQubeClient(sonarqube_url=sonar_url, token=sonar_token)

    def analyze_with_sonar(self, project_key):
        issues = self.sonar.issues.search_issues(componentKeys=project_key)
        results = []
        for issue in issues:
            results.append({
                "message": issue.get("message"),
                "severity": issue.get("severity"),
                "rule": issue.get("rule"),
                "line": issue.get("line")
            })
        return results

    def run(self, file_path, project_key="demo"):
        # 1. Load code
        with open(file_path, "r") as f:
            code = f.read()

        # 2. Run SonarQube analysis
        issues = self.analyze_with_sonar(project_key)

        if not issues:
            print(f"✅ No SonarQube issues found in {file_path}")
            return code

        # 3. Summarize issues
        issues_text = "\n".join(
            [f"- {i['severity']}: {i['message']} (rule: {i['rule']}, line {i['line']})"
             for i in issues]
        )

        # 4. Fix code with Hugging Face LLM
        prompt = f"""
        Here is Python code:\n{code}\n
        SonarQube found these issues:\n{issues_text}\n
        Please fix all issues and return the full corrected code.
        """
        fixed_code = hf_generate(prompt, 500)

        # 5. Replace file with improved code
        with open(file_path, "w") as f:
            f.write(fixed_code)

        print(f"🔧 Fixed {file_path} using SonarQube feedback.")
        return fixed_code


class AggregatorAgent(Agent):
    def run(self, ranked_solutions):
        best = ranked_solutions[0]
        return {
            "best_solution": best,
            "alternatives": ranked_solutions[1:]
        }
    

class ProgramTypeAgent(Agent):
    def run(self, file_name: str):
        prompt = f"""
        Identify the programming language from this file name: {file_name}.
        Respond with only the language name (e.g., Python, JavaScript, Java, TypeScript, C++).
        """
        return hf_generate(prompt, 20).strip()

# ==============================
# GitHub Repo Handler
# ==============================
class GitHubAPIHandler:
    def __init__(self, repo_url, token):
        # Parse repo_url like https://github.com/owner/repo.git
        parts = repo_url.rstrip(".git").split("/")
        self.owner, self.repo = parts[-2], parts[-1]
        self.base_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        self.headers = {"Authorization": f"token {token}"}

    def get_file_content(self, file_path):
        url = f"{self.base_url}/contents/{file_path}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]

    def update_file_content(self, file_path, new_content, sha, message="Update via Agentic AI"):
        url = f"{self.base_url}/contents/{file_path}"
        b64_content = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": message,
            "content": b64_content,
            "sha": sha
        }
        r = requests.put(url, headers=self.headers, json=payload)
        r.raise_for_status()
        return r.json()


# ==============================
# Pipeline
# ==============================
class AgenticPipeline:
    def __init__(self):
        self.dispatcher = DispatcherAgent()
        self.program_type = ProgramTypeAgent()
        self.suggester = CodeSuggestionAgent()
        self.ranker = RankingAgent()
        self.quality = CodeQualityAgent()
        self.aggregator = AggregatorAgent()

    async def  process(self, repo_url, file_path, prompt):
       
    
        #lang = self.program_type.run(full_path)
        #print(f"📝 Detected language: {lang}")

        repo_path=self.dispatcher.clone_repo(repo_url)
        full_path = os.path.join(repo_path,file_path.replace('/',"\\"))
        overallcontent= self.dispatcher.find_config_file(repo_path)
        mask_sensetive=self.dispatcher.mask_sensitive(overallcontent)
        analysis = await self.dispatcher.send_to_hf(mask_sensetive, "Please find language and version")
        language = analysis.get("language", "generic")
        lang=language["language"]
        return {"language":language}

        # Dispatcher decides
        #task_type = self.dispatcher.run(prompt)

        #if task_type == "code_suggestion":
        #    suggestions = self.suggester.run(prompt)
        #    ranked = self.ranker.run(suggestions)
        #    best = self.quality.run(full_path)   # Sonar + fix
        #    final = self.aggregator.run(ranked)
        #    final["best_solution"] = best
        #    return final

        #elif task_type == "code_quality":
        #    best = self.quality.run(full_path)   # Sonar + fix
        #    return {"best_solution": best, "alternatives": []}


# ==============================
# FastAPI App
# ==============================
app = FastAPI()

class CodeRequest(BaseModel):
    repo_url: str
    file_path: str
    prompt: str

class CodeResponse(BaseModel):
    status: str
    best_solution: str
    alternatives: list
    file_updated: str


pipeline = AgenticPipeline()

@app.post("/process", response_model=CodeResponse)
async def process_request(req: CodeRequest):
    result =await pipeline.process(req.repo_url, req.file_path, req.prompt)
    return result
    #return CodeResponse(
    #    status="success",
    #    best_solution=result.get("best_solution", ""),
    #    alternatives=result.get("alternatives", []),
    #    file_updated=req.file_path
    #)
