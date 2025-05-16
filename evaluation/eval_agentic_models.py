# Evaluate models on agentic final/raw datasets with JSON responses
import json
import re
import os
import argparse
import yaml
import csv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple, Union
from groq import Groq

openai_client = None
anthropic_client = None
gemini_client = None
groq_client = None


# Load dataset
def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# JSON format prompt templates
def build_json_prompt(task_id: str, sample: Dict) -> str:
    if task_id == "T2":
        context = sample.get("context")
        prompt = (
            "You are evaluating whether a paragraph has logically coherent sentence order.\n\n"
            "Paragraph:\n" + " ".join(context) + "\n\n"
            "Please provide your answer in the following JSON format:\n"
            "{\n"
            '  "answer": "yes" or "no"\n'
            "}\n\n"
            "Answer 'yes' if the sentences are in coherent order, 'no' if they are not."
        )
    elif task_id == "T3":
        sentence = sample.get("sentence", "")
        choices = sample.get("choices", [])
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(choices)])
        prompt = (
            "You need to identify which option is most anomalous or inconsistent for filling the blank.\n\n"
            f"Sentence: {sentence}\n\n"
            "Options:\n" + numbered + "\n\n"
            "Please provide your answer in the following JSON format:\n"
            "{\n"
            '  "answer": <number between 1 and 5>\n'
            "}\n\n"
            "Choose the most anomalous option."
        )
    elif task_id == "T4":
        paragraph_1 = sample.get("paragraph_1", [])
        paragraph_2 = sample.get("paragraph_2", [])
        bridges = sample.get("bridges", [])
        
        p1_text = " ".join(paragraph_1) if isinstance(paragraph_1, list) else paragraph_1
        p2_text = " ".join(paragraph_2) if isinstance(paragraph_2, list) else paragraph_2
        
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(bridges)])
        prompt = (
            "You need to identify which connecting sentence is most anomalous or inconsistent.\n\n"
            f"Paragraph 1: {p1_text}\n\n"
            f"Paragraph 2: {p2_text}\n\n"
            "Connecting sentence options:\n" + numbered + "\n\n"
            "Please provide your answer in the following JSON format:\n"
            "{\n"
            '  "answer": <number between 1 and 5>\n'
            "}\n\n"
            "Choose the most anomalous connecting sentence."
        )
    else:
        # T1, T5, T6, T7
        context = sample.get("context")
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(context)])
        prompt = (
            "You need to identify which sentence is most anomalous or inconsistent with the others.\n\n"
            "Sentences:\n" + numbered + "\n\n"
            "Please provide your answer in the following JSON format:\n"
            "{\n"
            '  "answer": <number between 1 and N>\n'
            "}\n\n"
            "Choose the most anomalous sentence."
        )
    
    return prompt

# Parse JSON response
def parse_json_response(response: str, task_id: str) -> Tuple[Union[int, bool, None], bool]:
    """Parse JSON response and return (answer, success)"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            # If no JSON found, treat the whole response as JSON
            data = json.loads(response)
        
        if task_id == "T2":
            answer = data.get("answer", "").lower()
            if answer == "yes":
                return True, True
            elif answer == "no":
                return False, True
            else:
                return None, False
        else:
            answer = data.get("answer")
            if isinstance(answer, int) and 1 <= answer <= 10:  # Allow up to 10 options
                return answer - 1, True  # Convert to 0-indexed
            else:
                return None, False
                
    except Exception as e:
        # Fallback to text parsing if JSON parsing fails
        try:
            if task_id == "T2":
                if "yes" in response.lower() and "no" not in response.lower():
                    return True, True
                elif "no" in response.lower() and "yes" not in response.lower():
                    return False, True
                else:
                    return None, False
            else:
                match = re.search(r'\b(\d+)\b', response)
                if match:
                    num = int(match.group(1))
                    if 1 <= num <= 10:
                        return num - 1, True
                    else:
                        return None, False
                else:
                    return None, False
        except Exception as fallback_e:
            return None, False

# OpenAI evaluation with JSON
def evaluate_sample_openai(sample: Dict, model: str) -> Tuple[Optional[bool], bool]:
    try:
        prompt = build_json_prompt(sample["task_id"], sample)

        if model.startswith("gpt-"):
            res = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4") else None
            )
        elif model.startswith("o"):
            res = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            
        output = res.choices[0].message.content.strip()
        
        answer, parsed = parse_json_response(output, sample["task_id"])
        if answer is None:
            return None, False
            
        if sample["task_id"] == "T2":
            expected = sample.get("is_coherent", False)
            return answer == expected, True
        else:
            return answer == sample["anomaly_index"], True
            
    except Exception as e:
        print(f"❌ {sample['sample_id']} | {model} | Error: {e}")
        return None, False

# Claude evaluation with JSON
def evaluate_sample_claude(sample: Dict, model: str) -> Tuple[Optional[bool], bool]:
    try:
        prompt = build_json_prompt(sample["task_id"], sample)
        res = anthropic_client.messages.create(
            model=model,
            max_tokens=100,  # Increased for JSON response
            messages=[{"role": "user", "content": prompt}]
        )
        output = res.content[0].text.strip()
        
        answer, parsed = parse_json_response(output, sample["task_id"])
        if answer is None:
            return None, False
            
        if sample["task_id"] == "T2":
            expected = sample.get("is_coherent", False)
            return answer == expected, True
        else:
            return answer == sample["anomaly_index"], True
            
    except Exception as e:
        print(f"❌ {sample['sample_id']} | {model} | Error: {e}")
        return None, False

# Gemini evaluation with JSON
def evaluate_sample_gemini(sample: Dict, model: str) -> Tuple[Optional[bool], bool]:
    try:
        prompt = build_json_prompt(sample["task_id"], sample)
        model_obj = genai.GenerativeModel(model)
        res = model_obj.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=100,
                response_mime_type="application/json"
            )
        )
        output = res.text.strip()
        
        answer, parsed = parse_json_response(output, sample["task_id"])
        if answer is None:
            return None, False
            
        if sample["task_id"] == "T2":
            expected = sample.get("is_coherent", False)
            return answer == expected, True
        else:
            return answer == sample["anomaly_index"], True
            
    except Exception as e:
        print(f"❌ {sample['sample_id']} | {model} | Error: {e}")
        return None, False

# Groq evaluation with JSON
def evaluate_sample_groq(sample: Dict, model: str) -> Tuple[Optional[bool], bool]:
    try:
        prompt = build_json_prompt(sample["task_id"], sample)
        res = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = res.choices[0].message.content.strip()
        
        answer, parsed = parse_json_response(output, sample["task_id"])
        if answer is None:
            return None, False
            
        if sample["task_id"] == "T2":
            expected = sample.get("is_coherent", False)
            return answer == expected, True
        else:
            return answer == sample["anomaly_index"], True
            
    except Exception as e:
        print(f"❌ {sample['sample_id']} | {model} | Error: {e}")
        return None, False

        
def evaluate_model_on_dataset(dataset: List[Dict], model: str, provider: str, results: List[Dict]):
    correct = 0
    parsed_successfully = 0
    total = len(dataset)
    print(f"\n🔍 Evaluating {model} ({provider}) on {total} samples...")
    
    for sample in dataset:
        if provider == "openai":
            result, parsed = evaluate_sample_openai(sample, model)
        elif provider == "claude":
            result, parsed = evaluate_sample_claude(sample, model)
        elif provider == "gemini":
            result, parsed = evaluate_sample_gemini(sample, model)
        elif provider == "groq":
            result, parsed = evaluate_sample_groq(sample, model)
        else:
            result, parsed = None, False
        
        # Handle None result
        if result is None:
            result = False
        
        results.append({
            "sample_id": sample["sample_id"],
            "task_id": sample["task_id"],
            "model": model,
            "provider": provider,
            "correct": result,
            "parsed": parsed
        })
        
        if result:
            correct += 1
        if parsed:
            parsed_successfully += 1

    acc = correct / total
    parse_rate = parsed_successfully / total
    print(f"✅ {model} Accuracy: {acc:.2%} ({correct}/{total})")
    print(f"📝 {model} Parse Rate: {parse_rate:.2%} ({parsed_successfully}/{total})")

# Save to CSV with parsing info
def save_results_to_csv(results: List[Dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "task_id", "model", "provider", "correct", "parsed"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📄 Saved results to {path}")

# Calculate detailed statistics
def calculate_detailed_stats(results: List[Dict]):
    task_stats = {}
    model_stats = {}
    
    for res in results:
        task_id = res["task_id"]
        model = res["model"]
        
        # Per-task stats
        if task_id not in task_stats:
            task_stats[task_id] = {"correct": 0, "total": 0, "parsed": 0}
        task_stats[task_id]["total"] += 1
        if res["correct"]:
            task_stats[task_id]["correct"] += 1
        if res["parsed"]:
            task_stats[task_id]["parsed"] += 1
        
        # Per-model-per-task stats
        key = (model, task_id)
        if key not in model_stats:
            model_stats[key] = {"correct": 0, "total": 0, "parsed": 0}
        model_stats[key]["total"] += 1
        if res["correct"]:
            model_stats[key]["correct"] += 1
        if res["parsed"]:
            model_stats[key]["parsed"] += 1
    
    print("\n📊 Per-task Statistics:")
    for task_id, stats in task_stats.items():
        acc = stats["correct"] / stats["total"] * 100
        parse_rate = stats["parsed"] / stats["total"] * 100
        print(f"  {task_id}: {acc:.1f}% accuracy, {parse_rate:.1f}% parse rate ({stats['correct']}/{stats['total']})")
    
    print("\n📊 Per-model-per-task Statistics:")
    for (model, task_id), stats in model_stats.items():
        acc = stats["correct"] / stats["total"] * 100
        parse_rate = stats["parsed"] / stats["total"] * 100
        print(f"  {model} - {task_id}: {acc:.1f}% accuracy, {parse_rate:.1f}% parse rate ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Path to agentic_final.jsonl or agentic_raw.jsonl")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    openai_api_key = cfg.get("openai_api_key")
    claude_api_key = cfg.get("claude_api_key")
    gemini_api_key = cfg.get("gemini_api_key")
    groq_api_key = cfg.get("groq_api_key")
    models = cfg.get("models", [])
    
    output_suffix = "final" if "final" in args.dataset else "raw"
    csv_output = f"evaluation_results_{output_suffix}_json.csv"

    openai_client = OpenAI(api_key=openai_api_key)
    anthropic_client = Anthropic(api_key=claude_api_key)
    genai.configure(api_key=gemini_api_key)
    groq_client = Groq(api_key=groq_api_key)

    dataset = load_dataset(args.dataset)
    all_results = []
    
    for m in models:
        evaluate_model_on_dataset(dataset, model=m["name"], provider=m["provider"], results=all_results)

    save_results_to_csv(all_results, csv_output)
    calculate_detailed_stats(all_results)