from flask import Flask, render_template, request, jsonify
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file
from datetime import datetime
import pandas as pd
import os
import json
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = Flask(__name__) 
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LAST_FILE = None
LAST_INSIGHTS = ""


def generate_with_retry(prompt):
    for i in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e):
                time.sleep(5)
            else:
                raise

    return "⚠️ Gemini AI is currently busy. Please try again in a minute."


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CSV LOADER ----------------
def load_csv(filepath):
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(",", expand=True)
        df.columns = [
            "Area",
            "Complaints",
            "Air_Quality",
            "Water_Usage",
            "Traffic_Level",
            "Waste_Collected"
        ]

    df.columns = df.columns.astype(str).str.strip()

    df["Complaints"] = pd.to_numeric(df["Complaints"], errors="coerce").fillna(0)
    df["Water_Usage"] = pd.to_numeric(df["Water_Usage"], errors="coerce").fillna(0)

    return df


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    global LAST_FILE

    file = request.files.get("file")
    if not file:
        return "No file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    LAST_FILE = filepath

    df = load_csv(filepath)
    highest = df.loc[df["Complaints"].idxmax()]

    return render_template(
        "dashboard.html",
        rows=len(df),
        columns=len(df.columns),
        highest_area=highest["Area"],
        highest_value=highest["Complaints"],
        avg_water=round(df["Water_Usage"].mean(), 2),

        areas=json.dumps(df["Area"].astype(str).tolist()),
        complaints=json.dumps(df["Complaints"].tolist()),
        water_usage=json.dumps(df["Water_Usage"].tolist()),

        air_labels=json.dumps(list(df["Air_Quality"].value_counts().index.astype(str))),
        air_values=json.dumps(list(df["Air_Quality"].value_counts().values.tolist()))
    )


# ---------------- SAFE MODEL CALL ----------------
def get_model():
    # stable fallback models
    try:
        return genai.GenerativeModel("gemini-1.5-pro")
    except:
        return genai.GenerativeModel("gemini-pro")


# ---------------- AI INSIGHTS ---------------
@app.route("/ai-insights", methods=["POST"])
def ai_insights():
    global LAST_FILE, LAST_INSIGHTS

    if not LAST_FILE:
        return jsonify({"insights": "No file uploaded yet"})

    try:
        df = load_csv(LAST_FILE)

        prompt = f"""
You are a city data analyst.

Analyze this dataset:

{df.to_string(index=False)}

Return:
- Top 3 problem areas
- Pollution hotspots
- Water usage insights
- Traffic issues
- 5 recommendations
"""

        try:
            text = generate_with_retry(prompt)

        except Exception as e:
            msg = str(e)

            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                text = (
                    "⚠️ AI service is temporarily busy because the free Gemini quota has been reached.\n\n"
                    "Please wait a few seconds and try again."
                )
            else:
                text = f"Error: {msg}"

        LAST_INSIGHTS = text

        return jsonify({"insights": text})

    except Exception as e:
        return jsonify({"insights": str(e)})


# ---------------- ASK AI ----------------
@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    global LAST_FILE

    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."})

    try:
        # Use uploaded dataset if available
        if LAST_FILE:
            df = load_csv(LAST_FILE)

            prompt = f"""
You are CommunitySense AI, a smart civic data assistant.

Dataset:
{df.to_string(index=False)}

Answer the following question based ONLY on this dataset.

Question:
{question}
"""
        else:
            prompt = f"""
You are CommunitySense AI.

Question:
{question}
"""

        text = generate_with_retry(prompt)

        return jsonify({"answer": text})

    except Exception as e:
        msg = str(e)

        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return jsonify({
                "answer": "⚠️ Gemini API quota reached. Please wait a few seconds and try again."
            })

        return jsonify({"answer": f"Error: {msg}"})

@app.route("/download-report")
def download_report():

    global LAST_FILE, LAST_INSIGHTS

    if not LAST_FILE:
        return "Upload a dataset first."

    df = load_csv(LAST_FILE)

    pdf_name = "CommunitySense_AI_Report.pdf"

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(pdf_name)

    story = []

    story.append(Paragraph("<b><font size=18>CommunitySense AI Report</font></b>", styles["Title"]))

    story.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Dataset Summary</b>", styles["Heading2"]))

    story.append(Paragraph(f"Rows: {len(df)}", styles["Normal"]))

    story.append(Paragraph(f"Columns: {len(df.columns)}", styles["Normal"]))

    story.append(Paragraph(f"Highest Complaints: {df['Complaints'].max()}", styles["Normal"]))

    story.append(Paragraph(f"Average Water Usage: {round(df['Water_Usage'].mean(),2)}", styles["Normal"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>AI Insights</b>", styles["Heading2"]))

    for line in LAST_INSIGHTS.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)

    return send_file(pdf_name, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)