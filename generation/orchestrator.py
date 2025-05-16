import json
import re
from typing import Dict, Tuple, Optional, Any

from utils import llm_call
from utils import extract_json
from utils import log_step  # 상단에 임포트 추가


# -- Evaluate by Orchestrator --
def orchestrator_check_init(task_id: str, sample: Dict[str, Any], model: str = "gpt-4o", is_final_attempt: bool = False, sample_index: int = 0) -> Tuple[bool, Optional[str]]:
    """최초 문제 생성 단계(init)에서 문제의 구조적 타당성을 검사하는 함수"""
    prompt = f"You are a benchmark quality controller evaluating if this problem is well-formed and structured correctly for task {task_id}.\n\n"
    
    # 평가 기준 설정 부분에 마지막 시도 여부 추가
    if is_final_attempt:
        prompt += "CRITICAL INSTRUCTION: This is the final attempt. Your primary goal is to APPROVE this problem unless it has FATAL flaws that make it completely unsolvable. Minor issues should be ignored. The problem only needs to be minimally functional - not perfect. If there's any reasonable way a student could solve this problem, you MUST approve it.\n\n"
    # if is_final_attempt:
    #     prompt += "Note: This is the final attempt. While maintaining quality standards, be MORE lenient in your evaluation. Accept problems that are reasonable and solvable, even if they have moderate imperfections.\n\n"
    # else:
    #     prompt += "Note: While maintaining quality standards, be lenient in your evaluation. Accept problems that are reasonable and solvable, even if they have minor imperfections.\n\n"
    
    task_names = {
        "T1": "Sentence Context Anomaly",
        "T2": "Paragraph Order Consistency",
        "T3": "Blank-based Choice Anomaly",
        "T4": "Bridge Sentence Evaluation",
        "T5": "Referential Ambiguity",
        "T6": "Logical Contradiction",
        "T7": "Tone/Style Violation"
    }
    task_name = task_names.get(task_id, "Anomaly Detection")
    
    prompt += f"Task Type: {task_name} ({task_id})\n\n"

    # 여기에 Task Description 추가
    task_descriptions = {
        "T1": "This task requires generating 5-6 sentences on a topic where one of them is anomalous (semantically inconsistent or conceptually off-topic). The anomaly should be detectable but not overly obvious, requiring careful reading to identify.",
        "T2": "This task requires creating 5 sentences about a topic, either in logically coherent order (is_coherent: true) or with subtly disrupted order (is_coherent: false). For the disrupted version, 1-2 sentences should be moved to disrupt temporal/causal flow while avoiding obvious scrambling.",
        "T3": "This task requires designing a sentence completion question with a blank (marked as ___) and 5 answer choices. One choice should be subtly inappropriate or anomalous. The anomaly should be detectable with careful analysis.",
        "T4": "This task requires creating two short paragraphs (2-3 sentences each) and 5 candidate bridge sentences to connect them. One bridge should be contextually or logically weak compared to the others.",
        "T5": "This task requires writing 5 sentences with pronouns and referents where one sentence contains ambiguous pronouns (unclear 'he', 'it', etc.). The ambiguity should be noticeable upon careful reading.",
        "T6": "This task requires generating 5 statements where one contains a logical contradiction or reversed logic. The contradiction should be subtle but detectable upon careful examination.",
        "T7": "This task requires writing 5 sentences with consistent formal/academic tone where one sentence subtly violates the established tone/style. The violation should be identifiable but not overly obvious."
    }
    
    task_structure = {
        "T1": "The expected JSON structure should include 'context' (array of 5-6 sentences), 'anomaly_index' (integer indicating which sentence is anomalous), and 'meta' (with source, topic, and anomaly_type).",
        "T2": "The expected JSON structure should include 'context' (array of 5 sentences) and 'is_coherent' (boolean indicating if the order is logical or disrupted).",
        "T3": "The expected JSON structure should include 'sentence' (with a blank marked as ___), 'choices' (array of 5 options), and 'anomaly_index' (integer indicating which choice is anomalous).",
        "T4": "The expected JSON structure should include 'paragraph_1' (array of sentences), 'paragraph_2' (array of sentences), 'bridges' (array of 5 bridge options), and 'anomaly_index' (integer indicating which bridge is weak).",
        "T5": "The expected JSON structure should include 'context' (array of 5 sentences) and 'anomaly_index' (integer indicating which sentence has ambiguous pronouns).",
        "T6": "The expected JSON structure should include 'context' (array of 5 statements) and 'anomaly_index' (integer indicating which statement has a contradiction).",
        "T7": "The expected JSON structure should include 'context' (array of 5 sentences) and 'anomaly_index' (integer indicating which sentence violates the tone/style)."
    }
    
    prompt += f"Task Description: {task_descriptions.get(task_id, '')}\n\n"
    prompt += f"Expected Structure: {task_structure.get(task_id, '')}\n\n"
    
    if task_id == "T2":
        passage = " ".join(sample["context"])
        if sample.get("is_coherent") == True:
            label = "True (coherent)"
        else:
            label = "False (incoherent)"
        prompt += f"Paragraph: {passage}\n\nCorrect Answer: {label}\n\n"
    
    elif task_id == "T3":
        sentence = sample.get("sentence", "")
        choices = sample.get("choices", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        prompt += f"Sentence: {sentence}\n\n"
        prompt += "Choices:\n" + "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    elif task_id == "T4":
        paragraph_1 = sample.get("paragraph_1", [])
        paragraph_2 = sample.get("paragraph_2", [])
        bridges = sample.get("bridges", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        p1_text = " ".join(paragraph_1) if isinstance(paragraph_1, list) else paragraph_1
        p2_text = " ".join(paragraph_2) if isinstance(paragraph_2, list) else paragraph_2
        
        prompt += f"Paragraph 1: {p1_text}\n\n"
        prompt += f"Paragraph 2: {p2_text}\n\n"
        prompt += "Bridge Options:\n" + "\n".join([f"{i+1}. {bridge}" for i, bridge in enumerate(bridges)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    else:  # T1, T5, T6, T7
        context = sample.get("context", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        prompt += "Context:\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(context)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    # 평가 기준 및 JSON 응답 요청
    prompt += "Note: While maintaining quality standards, be lenient in your evaluation. Accept problems that are reasonable and solvable, even if they have minor imperfections.\n\n"
    prompt += "Evaluate the problem based on these criteria:\n"
    prompt += "1. VALIDITY: Is the problem well-formed and complete?\n"
    prompt += "2. TYPE ADHERENCE: Does the problem follow the expected task type requirements?\n"
    prompt += "3. LOGICAL COHERENCE: Is the correct answer clearly identifiable?\n"
    prompt += "4. FAIRNESS: Is the problem fair and reasonable? Does it have a clear, unambiguous solution?\n\n"

    prompt += "Return your evaluation in JSON format:\n"
    prompt += "{\n"
    prompt += '  "approved": boolean (true if the problem passes all criteria, false otherwise),\n'
    prompt += '  "feedback": null if approved, or detailed feedback if rejected addressing:\n'
    prompt += '              - Problem construction issues\n'
    prompt += '              - Anomaly ambiguity concerns\n'
    prompt += '              - Specific improvement suggestions\n'
    prompt += "}"

    # 로깅: orchestrator 프롬프트
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="init",
        agent="orchestrator",
        action="validate_request",
        input_content=prompt,
        metadata={
            "is_final_attempt": is_final_attempt,
            "sample_id": sample.get("sample_id", "unknown")
        }
    )

    res = llm_call(prompt, model=model)

    # 로깅: orchestrator 응답
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="init",
        agent="orchestrator",
        action="validate_response",
        output_content=res,
        metadata={
            "model": model
        }
    )
    
    try:
        # JSON 파싱 시도
        result = extract_json(res)
        approved = result.get("approved", False)
        feedback = result.get("feedback")

        # 로깅: 파싱된 결과
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="init",
            agent="orchestrator",
            action="validation_result",
            output_content={
                "approved": approved,
                "feedback": feedback
            },
            metadata={}
        )

        return approved, feedback
    
    except Exception as e:
        # JSON 파싱 실패 시 기존 방식으로 폴백
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="init",
            agent="system",
            action="error",
            output_content=str(e),
            metadata={
                "error_type": "json_parsing",
                "raw_response": res
            }
        )

        print(f"Warning: Failed to parse JSON response: {e}")
        if "approve" in res.lower() and not "reject" in res.lower():
            return True, None
        else:
            if "reject:" in res.lower():
                feedback = res.split("Reject:", 1)[1].strip()
            else:
                feedback = res
            return False, feedback



def orchestrator_get_feedback(task_id: str, sample: Dict[str, Any], student_explanation: str, model: str = "gpt-4o", sample_index: int = 0) -> str:
    """학생이 문제를 맞췄을 때, 난이도를 올리기 위한 피드백을 생성하는 함수"""
    prompt = f"You are helping to create a harder version of a problem that a student has correctly solved. Analyze the student's solution and provide feedback.\n\n"
    
    task_names = {
        "T1": "Sentence Context Anomaly",
        "T2": "Paragraph Order Consistency",
        "T3": "Blank-based Choice Anomaly",
        "T4": "Bridge Sentence Evaluation",
        "T5": "Referential Ambiguity",
        "T6": "Logical Contradiction",
        "T7": "Tone/Style Violation"
    }
    task_name = task_names.get(task_id, "Anomaly Detection")
    difficulty = sample.get("meta", {}).get("difficulty_level", "unknown")
    
    prompt += f"Task Type: {task_name} ({task_id})\n"
    prompt += f"Current Difficulty: {difficulty}\n\n"
    
    # 전체 문제를 포함
    prompt += f"ORIGINAL PROBLEM:\n{json.dumps(sample, ensure_ascii=False, indent=2)}\n\n"
    
    # 문제의 핵심 요소만 간략히 포함
    if task_id == "T2":
        prompt += f"Problem Type: Paragraph coherence assessment\n"
        prompt += f"Current Answer: {'Coherent' if sample.get('is_coherent') else 'Not Coherent'}\n\n"
    elif task_id == "T3":
        prompt += f"Problem Type: Sentence completion with anomalous option\n"
        prompt += f"Anomaly Index: {sample.get('anomaly_index')}\n\n"
    elif task_id == "T4":
        prompt += f"Problem Type: Bridge sentence identification\n"
        prompt += f"Anomaly Index: {sample.get('anomaly_index')}\n\n"
    else:
        prompt += f"Problem Type: Anomaly detection in context\n"
        prompt += f"Anomaly Index: {sample.get('anomaly_index')}\n\n"
      
    prompt += f"Student's Explanation: {student_explanation}\n\n"
    
    prompt += "Based on how the student solved this problem, provide feedback to create a more challenging version:\n"
    prompt += "1. What aspects did the student easily identify?\n"
    prompt += "2. How could the problem be made more subtle or complex?\n"
    prompt += "3. Give specific suggestions for increasing difficulty.\n\n"
    
    prompt += "Return your feedback in JSON format:\n"
    prompt += "{\n"
    prompt += '  "analysis": "Brief analysis of student solution",\n'
    prompt += '  "suggestions": ["Specific suggestion 1", "Specific suggestion 2", ...],\n'
    prompt += '  "difficulty_increase": "Summary of how to increase difficulty"\n'
    prompt += "}"

    # 로깅: orchestrator 피드백 요청
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="difficulty_increase",
        agent="orchestrator",
        action="feedback_request",
        input_content=prompt,
        metadata={
            "difficulty": difficulty,
            "sample_id": sample.get("sample_id", "unknown")
        }
    )

    res = llm_call(prompt, model=model)
    
    # 로깅: orchestrator 피드백 응답
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="difficulty_increase",
        agent="orchestrator",
        action="feedback_response",
        output_content=res,
        metadata={
            "model": model
        }
    )

    try:
        # JSON 파싱 시도
        result = extract_json(res)
        
        # 피드백 구성
        feedback = result.get("analysis", "") + "\n\n"
        
        if "suggestions" in result and isinstance(result["suggestions"], list):
            feedback += "Suggestions:\n- " + "\n- ".join(result["suggestions"]) + "\n\n"
        
        feedback += result.get("difficulty_increase", "")

        # 로깅: 파싱된 피드백
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="difficulty_increase",
            agent="orchestrator",
            action="feedback_parsed",
            output_content=feedback,
            metadata={
                "has_suggestions": "suggestions" in result and len(result.get("suggestions", [])) > 0
            }
        )

        return feedback
    except Exception as e:
        # JSON 파싱 실패 시 원본 응답 반환
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="difficulty_increase",
            agent="system",
            action="error",
            output_content=str(e),
            metadata={
                "error_type": "json_parsing",
                "raw_response": res
            }
        )

        print(f"Warning: Failed to parse feedback JSON: {e}")
        return res


def orchestrator_check_problem(task_id: str, sample: Dict[str, Any], model: str = "gpt-4o", sample_index: int = 0) -> Tuple[bool, Optional[str]]:
    """난이도 증가 후 생성된 문제의 품질을 검증하는 함수"""
    # 기본적으로 check_init과 동일한 검증을 수행하지만, 난이도 정보를 추가
    prompt = f"You are a benchmark quality controller evaluating if a problem with increased difficulty is well-formed and appropriate for task {task_id}.\n\n"
    
    task_names = {
        "T1": "Sentence Context Anomaly",
        "T2": "Paragraph Order Consistency",
        "T3": "Blank-based Choice Anomaly",
        "T4": "Bridge Sentence Evaluation",
        "T5": "Referential Ambiguity",
        "T6": "Logical Contradiction",
        "T7": "Tone/Style Violation"
    }
    task_name = task_names.get(task_id, "Anomaly Detection")
    difficulty = sample.get("meta", {}).get("difficulty_level", "unknown")
    
    prompt += f"Task Type: {task_name} ({task_id})\n"
    prompt += f"Difficulty Level: {difficulty}\n\n"

    # 여기에 Task Description 추가
    task_descriptions = {
        "T1": "This task requires generating 5-6 sentences on a topic where one of them is anomalous (semantically inconsistent or conceptually off-topic). The anomaly should be detectable but not overly obvious, requiring careful reading to identify.",
        "T2": "This task requires creating 5 sentences about a topic, either in logically coherent order (is_coherent: true) or with subtly disrupted order (is_coherent: false). For the disrupted version, 1-2 sentences should be moved to disrupt temporal/causal flow while avoiding obvious scrambling.",
        "T3": "This task requires designing a sentence completion question with a blank (marked as ___) and 5 answer choices. One choice should be subtly inappropriate or anomalous. The anomaly should be detectable with careful analysis.",
        "T4": "This task requires creating two short paragraphs (2-3 sentences each) and 5 candidate bridge sentences to connect them. One bridge should be contextually or logically weak compared to the others.",
        "T5": "This task requires writing 5 sentences with pronouns and referents where one sentence contains ambiguous pronouns (unclear 'he', 'it', etc.). The ambiguity should be noticeable upon careful reading.",
        "T6": "This task requires generating 5 statements where one contains a logical contradiction or reversed logic. The contradiction should be subtle but detectable upon careful examination.",
        "T7": "This task requires writing 5 sentences with consistent formal/academic tone where one sentence subtly violates the established tone/style. The violation should be identifiable but not overly obvious."
    }
    
    task_structure = {
        "T1": "The expected JSON structure should include 'context' (array of 5-6 sentences), 'anomaly_index' (integer indicating which sentence is anomalous), and 'meta' (with source, topic, and anomaly_type).",
        "T2": "The expected JSON structure should include 'context' (array of 5 sentences) and 'is_coherent' (boolean indicating if the order is logical or disrupted).",
        "T3": "The expected JSON structure should include 'sentence' (with a blank marked as ___), 'choices' (array of 5 options), and 'anomaly_index' (integer indicating which choice is anomalous).",
        "T4": "The expected JSON structure should include 'paragraph_1' (array of sentences), 'paragraph_2' (array of sentences), 'bridges' (array of 5 bridge options), and 'anomaly_index' (integer indicating which bridge is weak).",
        "T5": "The expected JSON structure should include 'context' (array of 5 sentences) and 'anomaly_index' (integer indicating which sentence has ambiguous pronouns).",
        "T6": "The expected JSON structure should include 'context' (array of 5 statements) and 'anomaly_index' (integer indicating which statement has a contradiction).",
        "T7": "The expected JSON structure should include 'context' (array of 5 sentences) and 'anomaly_index' (integer indicating which sentence violates the tone/style)."
    }
    
    prompt += f"Task Description: {task_descriptions.get(task_id, '')}\n\n"
    prompt += f"Expected Structure: {task_structure.get(task_id, '')}\n\n"

    # Task별 문제 내용 포맷팅 (check_init과 동일)    
    if task_id == "T2":
        passage = " ".join(sample["context"])
        # 불리언 값을 명확하게 매핑
        if sample["is_coherent"] == True:  # 명시적 비교
            label = "True (coherent)"
        else:
            label = "False (incoherent)"
        prompt += f"Paragraph: {passage}\n\nCorrect Answer: {label}\n\n"
    
    elif task_id == "T3":
        sentence = sample.get("sentence", "")
        choices = sample.get("choices", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        prompt += f"Sentence: {sentence}\n\n"
        prompt += "Choices:\n" + "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    elif task_id == "T4":
        paragraph_1 = sample.get("paragraph_1", [])
        paragraph_2 = sample.get("paragraph_2", [])
        bridges = sample.get("bridges", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        p1_text = " ".join(paragraph_1) if isinstance(paragraph_1, list) else paragraph_1
        p2_text = " ".join(paragraph_2) if isinstance(paragraph_2, list) else paragraph_2
        
        prompt += f"Paragraph 1: {p1_text}\n\n"
        prompt += f"Paragraph 2: {p2_text}\n\n"
        prompt += "Bridge Options:\n" + "\n".join([f"{i+1}. {bridge}" for i, bridge in enumerate(bridges)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    else:  # T1, T5, T6, T7
        context = sample.get("context", [])
        anomaly_index = sample.get("anomaly_index", -1)
        
        prompt += "Context:\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(context)])
        prompt += f"\n\nCorrect Answer: Option {anomaly_index + 1}\n\n"
    
    # 평가 기준에 난이도 적절성 추가
    prompt += "Note: While maintaining quality standards, be lenient in your evaluation. Accept problems that are reasonable and solvable, even if they have minor imperfections.\n\n"
    prompt += "Evaluate the problem based on these criteria:\n"
    prompt += "1. VALIDITY: Is the problem well-formed and complete?\n"
    prompt += "2. TYPE ADHERENCE: Does the problem follow the expected task type requirements?\n"
    prompt += "3. LOGICAL COHERENCE: Is the correct answer clearly identifiable?\n"
    prompt += "4. FAIRNESS: Is the problem fair and reasonable? Does it have a clear, unambiguous solution?\n"
    prompt += f"5. DIFFICULTY: Is the difficulty appropriate for {difficulty} level?\n\n"

    prompt += "Return your evaluation in JSON format:\n"
    prompt += "{\n"
    prompt += '  "approved": boolean (true if the problem passes all criteria, false otherwise),\n'
    prompt += '  "feedback": null if approved, or detailed feedback if rejected addressing:\n'
    prompt += '              - Problem construction issues\n'
    prompt += '              - Anomaly ambiguity concerns\n'
    prompt += '              - Difficulty appropriateness\n'
    prompt += '              - Specific improvement suggestions\n'
    prompt += "}"

    # 로깅: orchestrator 난이도 증가 검증 요청
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="difficulty_increase",
        agent="orchestrator",
        action="validate_difficult",
        input_content=prompt,
        metadata={
            "difficulty": difficulty,
            "sample_id": sample.get("sample_id", "unknown")
        }
    )

    res = llm_call(prompt, model=model)
    
    # 로깅: orchestrator 난이도 증가 검증 응답
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="difficulty_increase",
        agent="orchestrator",
        action="validate_difficult_response",
        output_content=res,
        metadata={
            "model": model
        }
    )

    try:
        # JSON 파싱 시도
        result = extract_json(res)
        approved = result.get("approved", False)
        feedback = result.get("feedback")

        # 로깅: 파싱된 결과
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="difficulty_increase",
            agent="orchestrator",
            action="validation_difficult_result",
            output_content={
                "approved": approved,
                "feedback": feedback
            },
            metadata={}
        )
        
        return approved, feedback
    
    except Exception as e:
        # JSON 파싱 실패 시 기존 방식으로 폴백
        log_step(
            task_id=task_id,
            sample_index=sample_index,
            phase="difficulty_increase",
            agent="system",
            action="error",
            output_content=str(e),
            metadata={
                "error_type": "json_parsing",
                "raw_response": res[:100] + "..." if len(res) > 100 else res
            }
        )
        
        print(f"Warning: Failed to parse JSON response: {e}")
        if "approve" in res.lower() and not "reject" in res.lower():
            return True, None
        else:
            if "reject:" in res.lower():
                feedback = res.split("Reject:", 1)[1].strip()
            else:
                feedback = res
            return False, feedback

