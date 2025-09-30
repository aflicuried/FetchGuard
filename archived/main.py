from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class LoginForm(BaseModel): #被识别为body类型
    username: str
    password: str
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/text")
async def hello_text():
    return "Hello text!"

@app.get("/json")
async def hello_json():
    return {"message": "Hello json!"}

@app.get("/user/{username}")
def user_page(username: str):
    return f"Username: {username}"

@app.get("/search")
async def search_page(keyword: str = ""):
    return f"Keyword: {keyword}"

@app.post("/login") #loginForm因为BaseModel被识别为body
async def login(body: LoginForm):
    username = body.username
    password = body.password
    if username == "admin" or password == "123456":
        token = {"status": "login", "username": "admin"}
        return token
    else:
        return {"error": "Error"}
