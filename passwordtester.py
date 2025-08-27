from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# Password strength evaluation logic

def evaluate_password_strength(password):
    length = len(password)
    lower = bool(re.search(r"[a-z]", password))
    upper = bool(re.search(r"[A-Z]", password))
    digit = bool(re.search(r"\d", password))
    special = bool(re.search(r"[^A-Za-z0-9]", password))
    common = password.lower() in COMMON_PASSWORDS

    score = 0
    tips = []
    emoji = ""

    if length >= 12:
        score += 2
    elif length >= 8:
        score += 1
    else:
        tips.append("Use at least 12 characters for best security.")

    if lower:
        score += 1
    else:
        tips.append("Add lowercase letters.")
    if upper:
        score += 1
    else:
        tips.append("Add uppercase letters.")
    if digit:
        score += 1
    else:
        tips.append("Add numbers.")
    if special:
        score += 1
    else:
        tips.append("Add special characters (e.g. !@#$%).")
    if common:
        tips.append("Avoid common passwords!")
        score = 0

    # Creative feedback
    if score <= 1:
        verdict = "Very Weak"
        emoji = "😱"
    elif score == 2:
        verdict = "Weak"
        emoji = "😬"
    elif score == 3:
        verdict = "Medium"
        emoji = "🙂"
    elif score == 4:
        verdict = "Strong"
        emoji = "💪"
    elif score >= 5:
        verdict = "Very Strong"
        emoji = "🦾"
    else:
        verdict = "Unknown"
        emoji = "❓"

    return {
        "score": score,
        "verdict": verdict,
        "emoji": emoji,
        "tips": tips
    }

# A small set of common passwords for demo purposes
COMMON_PASSWORDS = set([
    "password", "123456", "123456789", "qwerty", "abc123", "password1", "111111", "123123"
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check_password():
    data = request.get_json()
    password = data.get("password", "")
    result = evaluate_password_strength(password)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5030)
