import json
import time
import asyncio
import os
from typing import Dict, Any, Tuple
import sqlite3
from sqlite3 import Error

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from passlib.context import CryptContext
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MAX_SCORE_PER_QUESTION = 80
MAX_TIME_SCORE = 20
MAX_CORRECTNESS_SCORE = 60
TOTAL_QUESTIONS = 5

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

users: Dict[str, Dict[str, Any]] = {}


# Database functions
def create_connection():
    try:
        conn = sqlite3.connect('leaderboard.db')
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
    return None

def create_table(conn):
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                score INTEGER NOT NULL,
                duration REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    except Error as e:
        print(f"Table creation error: {e}")

conn = create_connection()
if conn:
    create_table(conn)
    conn.close()
else:
    print("Error! Cannot create the database connection.")


# Helper functions
async def generate_python_questions():
    prompt = f"""Generate {TOTAL_QUESTIONS} Python coding questions in the following JSON format:
    [
        {{
            "id": 0,
            "question": "Write a Python function to...",
            "function_start": "def function_name(parameters):\\n    # Your code here\\n    ",
            "test_cases": [
                {{"input": "...", "expected_output": "..."}},
                {{"input": "...", "expected_output": "..."}},
                {{"input": "...", "expected_output": "..."}}
            ]
        }},
        ...
    ]
    Make sure the questions are of varying difficulty and cover different Python concepts."""

    try:
        print("Sending request to OpenAI API...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python coding questions."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        print(f"Raw API response content: {content}")
        content = content.strip().lstrip("```json").rstrip("```")
        questions = json.loads(content)
        print(f"Parsed questions: {json.dumps(questions, indent=2)}")
        return questions
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from OpenAI API: {e}")
        print(f"Response content: {content}")
    except Exception as e:
        print(f"Unexpected error generating questions: {e}")
    
    print("Returning fallback questions")
    return [
        {
            "id": 0,
            "question": "Write a Python function to calculate the area of a circle given its radius.",
            "function_start": "def calculate_circle_area(radius):\n    # Your code here\n    ",
            "test_cases": [
                {"input": "5", "expected_output": "78.54"},
                {"input": "0", "expected_output": "0"},
                {"input": "10", "expected_output": "314.16"}
            ]
        },
        # Add more fallback questions here
    ]

async def evaluate_code(code: str, question: Dict[str, Any]) -> Tuple[int, str]:
    prompt = f"""Evaluate the following Python code for the given question:

Question: {question['question']}

User's Code:
{code}

Test Cases:
{json.dumps(question['test_cases'], indent=2)}

Please evaluate the code based on the following criteria:
1. Correctness: Does it pass all test cases? (Max score: 40)
2. Efficiency: Is the solution optimized? (Max score: 10)
3. Code Quality: Is the code well-written, following Python best practices? (Max score: 10)

Provide a score out of 60 and a brief explanation of the evaluation.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python code evaluator. Provide honest and constructive feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        evaluation = response.choices[0].message.content
        score = int(evaluation.split('Score:')[1].split('/')[0].strip())
        return score, evaluation
    except Exception as e:
        print(f"Error evaluating code: {e}")
        return 0, "Error occurred during evaluation."
    


# API routes
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Username already exists"})
    hashed_password = pwd_context.hash(password)
    users[username] = {
        "password": hashed_password,
        "start_time": None,
        "answers": {},
        "current_question": 0,
        "question_start_time": None,
        "current_code": "",
        "total_score": 0
    }
    return templates.TemplateResponse("index.html", {"request": request, "message": "Registration successful. Please log in."})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username not in users or not pwd_context.verify(password, users[username]["password"]):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Incorrect username or password"})
    
    users[username]["start_time"] = time.time()
    users[username]["question_start_time"] = time.time()
    
    print(f"Generating questions for user: {username}")
    questions = await generate_python_questions()
    print(f"Generated {len(questions)} questions")
    users[username]["questions"] = questions
    users[username]["current_question"] = 0
    
    return templates.TemplateResponse("question.html", {
        "request": request,
        "username": username,
        "question": questions[0],
        "question_number": 1,
        "total_questions": len(questions),
        "is_last_question": False
    })

@app.post("/submit-code")
async def submit_code(request: Request, username: str = Form(...), question_id: int = Form(...), code: str = Form(...), score: int = Form(...)):
    if username not in users:
        raise HTTPException(status_code=400, detail="User not logged in")
    
    user_data = users[username]
    
    if question_id < 0 or question_id >= len(user_data["questions"]):
        raise HTTPException(status_code=400, detail=f"Invalid question ID: {question_id}. Total questions: {len(user_data['questions'])}")
    
    user_data["answers"][question_id] = {
        "code": code,
        "time_score": score,
    }
    
    next_question_id = question_id + 1
    user_data["current_question"] = next_question_id
    
    if next_question_id < len(user_data["questions"]):
        next_question = user_data["questions"][next_question_id]
        user_data["question_start_time"] = time.time()
        user_data["current_code"] = ""
        
        return templates.TemplateResponse("question.html", {
            "request": request,
            "username": username,
            "question": next_question,
            "question_number": next_question_id + 1,
            "total_questions": len(user_data["questions"]),
            "is_last_question": next_question_id == len(user_data["questions"]) - 1
        })
    else:
        raise HTTPException(status_code=400, detail="Quiz already completed")

@app.post("/update-current-code")
async def update_current_code(username: str = Form(...), code: str = Form(...), score: int = Form(...)):
    if username in users:
        users[username]["current_code"] = code
        users[username]["current_score"] = score
    return {"status": "ok"}

@app.post("/finish-quiz")
async def finish_quiz(request: Request, username: str = Form(...), code: str = Form(...), score: int = Form(...)):
    if username not in users:
        raise HTTPException(status_code=400, detail="User not logged in")
    
    user_data = users[username]
    end_time = time.time()
    duration = end_time - user_data["start_time"]
    
    last_question_id = len(user_data["questions"]) - 1
    user_data["answers"][last_question_id] = {
        "code": code,
        "time_score": score,
    }
    
    evaluation_tasks = [evaluate_code(answer["code"], user_data["questions"][question_id]) 
                        for question_id, answer in user_data["answers"].items()]
    
    evaluation_results = await asyncio.gather(*evaluation_tasks)
    
    total_score = 0
    for question_id, (correctness_score, evaluation) in enumerate(evaluation_results):
        answer = user_data["answers"][question_id]
        answer["correctness_score"] = correctness_score
        answer["total_score"] = answer["time_score"] + correctness_score
        answer["evaluation"] = evaluation
        total_score += answer["total_score"]
    
    user_data["total_score"] = total_score
    
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO leaderboard (username, score, duration) VALUES (?, ?, ?)",
                    (username, total_score, duration))
        conn.commit()
        conn.close()
    
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "username": username, 
        "score": total_score, 
        "max_possible_score": len(user_data["questions"]) * MAX_SCORE_PER_QUESTION,
        "duration": duration,
        "total_questions": len(user_data["questions"]),
        "MAX_SCORE_PER_QUESTION": MAX_SCORE_PER_QUESTION,
        "answers": user_data["answers"]
    })

@app.get("/leaderboard-data", response_class=HTMLResponse)
async def get_leaderboard_data(request: Request):
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT username, score, duration FROM leaderboard ORDER BY score DESC, duration ASC LIMIT 10")
        leaderboard_data = cur.fetchall()
        conn.close()
        
        leaderboard = [{"username": row[0], "score": row[1], "duration": row[2]} for row in leaderboard_data]
        
        return templates.TemplateResponse("leaderboard_partial.html", {"request": request, "leaderboard": leaderboard})
    else:
        return templates.TemplateResponse("leaderboard_partial.html", {"request": request, "leaderboard": []})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)