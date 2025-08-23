# -*- coding: utf-8 -*-
"""
Cybersecurity Threat Detection + Response Pipeline
-------------------------------------------------
1. Detection Agent: Analyzes network/system logs.
2. Response Agent: Uses Gemini 2.5 Flash API to decide actions.

Dependencies:
    pip install pandas google-genai python-dotenv
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

# ================================================================
# Load API key from .env
# ================================================================
load_dotenv()  # loads variables from .env into environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. "
                     "Make sure your .env contains: GEMINI_API_KEY=your_key_here")


# ================================================================
# Gemini Response Agent
# ================================================================
class GeminiResponseAgent:
    def __init__(self, org_policies=None, api_key=None):
        if not api_key:
            raise ValueError("API key must be provided for Gemini client")
        self.client = genai.Client(api_key=api_key)
        self.org_policies = org_policies or "Default: contain first, then investigate."

    def decide_response(self, decision_output: dict) -> dict:
        prompt = f"""
You are a cybersecurity response agent.
Organization policies: {self.org_policies}

Threat details: {json.dumps(decision_output, indent=2)}

Based on the above, select your response:
- block_ip
- isolate_host
- alert_admin
- schedule_scan
- no_action

Return ONLY valid JSON in this format:
{{
  "action": "<action>",
  "target": "<ip/host>",
  "priority": "<low|medium|high>",
  "explanation": "<short reason>"
}}
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(thinking_budget=512)
            )
        )
        try:
            raw_text = response.candidates[0].content.parts[0].text
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json", "", 1).strip()
            return json.loads(cleaned)
        except Exception as e:
            return {
                "action": "alert_admin",
                "target": decision_output.get("target_host", "unknown"),
                "priority": "high",
                "explanation": f"Gemini parsing error: {e}, raw={getattr(response,'candidates',response)}"
            }


# ================================================================
# Detection Agent (dummy example)
# ================================================================
def run_detection(df: pd.DataFrame):
    decisions = []
    for _, row in df.iterrows():
        if row["Label"] != "BENIGN":
            decision = {
                "alert": 1,
                "threat_type": row["Label"],
                "confidence": 0.92,  # dummy score
                "source_ip": row.get("Source IP", "unknown"),
                "destination_port": row.get("Destination Port", "unknown"),
                "target_host": row.get("Target Host", "unknown"),
                "time": row.get("timestamp", "N/A")
            }
            decisions.append(decision)
    return decisions


# ================================================================
# Orchestration
# ================================================================
def process_detection_results(df, response_agent):
    detections = run_detection(df)
    for d in detections:
        plan = response_agent.decide_response(d)
        print("\n=== Threat Detected ===")
        print(json.dumps(d, indent=2))
        print("--- Response Plan ---")
        print(json.dumps(plan, indent=2))


# ================================================================
# Main Execution
# ================================================================
if __name__ == "__main__":
    # Example dataset (replace with real logs)
    df = pd.DataFrame([
        {"Label": "BENIGN", "Destination Port": 443, "Source IP": "10.0.0.5"},
        {"Label": "PortScan", "Destination Port": 22, "Source IP": "192.168.1.55"},
        {"Label": "DDoS", "Destination Port": 80, "Source IP": "203.0.113.45"}
    ])

    response_agent = GeminiResponseAgent(
        org_policies="Block port scans immediately, isolate compromised hosts, always notify SOC.",
        api_key=api_key
    )

    process_detection_results(df, response_agent)
