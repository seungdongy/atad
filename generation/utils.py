
import json
import re
from openai import OpenAI
from typing import Dict, Any
import datetime
import copy
from groq import Groq

groq_api_key = "Your_API_KEY"
groq_client = Groq(api_key=groq_api_key)

client = OpenAI(api_key="Your_API_KEY")
claude_api_key = "Your_API_KEY"
gemini_api_key = "Your_API_KEY"

# -- Call LLM
def gpt_call(prompt: str, model: str = "gpt-4o") -> str:
    """OpenAI GPT 모델 호출 함수"""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 호출 오류: {e}")
        raise

# Claude 모델용 함수
def claude_call(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """Anthropic Claude 모델 호출 함수"""
    try:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        response = claude_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.content[0].text
    except ImportError:
        print("Error: anthropic 패키지가 설치되지 않았습니다.")
        raise
    except Exception as e:
        print(f"Claude 호출 오류: {e}")
        raise

# Gemini 모델용 함수
def gemini_call(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Google Gemini 모델 호출 함수"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt, generation_config={"temperature": 0.7})
        return response.text
    except ImportError:
        print("Error: google-generativeai 패키지가 설치되지 않았습니다.")
        raise
    except Exception as e:
        print(f"Gemini 호출 오류: {e}")
        raise

# Grok 모델용 함수
def grok_call(prompt: str, model: str = "grok-3") -> str:
    """xAI Grok 모델 호출 함수"""
    try:
        grok_client = OpenAI(
            api_key="ah-jik-ahn-ham-grok-api-key-here",  # xAI API 키
            base_url="https://api.x.ai/v1"     # xAI API 엔드포인트
        )
        
        response = grok_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Grok 호출 오류: {e}")
        raise

# LLaMa 호출
def groq_call(prompt: str, model: str = "llama-3.3-7b-versatile") -> str:
    """Groq API LLaMa 모델 호출 함수"""
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq 호출 오류: {e}")
        raise

# -- 통합 LLM 호출 함수 --
def llm_call(prompt: str, model: str = "gpt-4o") -> str:
    """다양한 LLM 모델 호출을 위한 통합 함수"""
    if model.startswith("claude"):
        return claude_call(prompt, model)
    elif model.startswith("gemini"):
        return gemini_call(prompt, model)
    elif model.startswith("grok"):
        return grok_call(prompt, model)
    elif model.startswith("llama"):
        return groq_call(prompt, model)
    else:  # GPT 모델들
        return gpt_call(prompt, model)


def extract_json(text: str) -> Dict[str, Any]:
    """
    문자열에서 JSON 객체를 추출하고 정리합니다.
    1. 코드 블록 내의 JSON을 찾습니다.
    2. 중괄호로 시작하고 끝나는 완전한 JSON 객체를 찾습니다.
    3. 제어 문자를 제거하고 JSON을 정리합니다.
    """
    try:
        # 코드 블록 내의 JSON 찾기
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # 제어 문자 처리
            json_str = re.sub(r'[\x00-\x1F\x7F]', ' ', json_str)
            return json.loads(json_str)
        
        # 첫 번째 완전한 JSON 객체 찾기
        start = text.find('{')
        if start >= 0:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 완전한 JSON 객체를 찾음
                        json_str = text[start:i+1]
                        # 제어 문자 처리
                        json_str = re.sub(r'[\x00-\x1F\x7F]', ' ', json_str)
                        return json.loads(json_str)
        
        # 마지막 시도: 전체 텍스트를 JSON으로 파싱
        # 제어 문자 처리
        clean_text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
        return json.loads(clean_text)

    except (json.JSONDecodeError, TypeError) as e:
        print(f"🛑 JSON parsing error: {e}")
        print(f"🧪 Raw text (first 500 chars):\n{text[:500]}")
        
        # 특별 케이스: JSON 파싱 실패 시 추가 시도
        try:
            # 문자열 불필요 문자 정리 시도
            if isinstance(text, str):
                # 제어 문자 모두 제거
                clean_text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
                # 따옴표 이스케이프 확인
                clean_text = clean_text.replace('\\"', '"').replace('\\\'', '\'')
                
                # JSON 형식 찾기 시도
                match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
                if match:
                    potential_json = match.group(1)
                    return json.loads(potential_json)
        except:
            pass
        
        raise

# -- Round-robin generator for topics/styles --
def round_robin(items):
    while True:
        for item in items:
            yield item


# 전역 로그 저장소
process_logs = []

def log_step(task_id, sample_index, phase, agent, action, input_content=None, output_content=None, metadata=None):
    """프로세스 로그를 저장하는 함수
    
    Args:
        task_id: 태스크 ID (T1, T2 등)
        sample_index: 샘플 인덱스 (루프 변수 i)
        phase: 단계 (init, student_evaluation, difficulty_increase 등)
        agent: 에이전트 (teacher, student, orchestrator, system 등)
        action: 액션 (prompt, response, validate 등)
        input_content: 입력 내용 (프롬프트 등)
        output_content: 출력 내용 (응답 등)
        metadata: 추가 메타데이터
    """

    timestamp = datetime.datetime.now().isoformat()

    # 객체의 깊은 복사 생성
    if isinstance(input_content, dict):
        input_content = copy.deepcopy(input_content)
    if isinstance(output_content, dict):
        output_content = copy.deepcopy(output_content)
    if isinstance(metadata, dict):
        metadata = copy.deepcopy(metadata)
        
    log_entry = {
        "timestamp": timestamp,
        "task_id": task_id,
        "sample_index": sample_index,
        "phase": phase,
        "agent": agent,
        "action": action,
        "input": input_content,
        "output": output_content,
        "metadata": metadata or {}
    }
    process_logs.append(log_entry)

def get_logs():
    """저장된 모든 로그를 반환"""
    return process_logs.copy()

def clear_logs():
    """로그 저장소 초기화"""
    process_logs.clear()