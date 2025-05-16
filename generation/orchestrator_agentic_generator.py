# ✅ Orchestrator-Aware Agentic Generator with Teacher-Student-Feedback Loop (Full Pipeline)
import random
import json
import re
import argparse
import yaml
import datetime
from openai import OpenAI
from itertools import cycle
from typing import List, Dict, Tuple, Optional, Any
from utils import llm_call, extract_json, round_robin, log_step, get_logs, clear_logs

from prompt_templates import build_teacher_prompt
from tasks_config import TASKS
from orchestrator import orchestrator_check_init, orchestrator_check_problem, orchestrator_get_feedback


# -- Evaluate Student answer --
def student_answer_with_context(task_id: str, sample: Dict[str, Any], context: List[Dict[str, Any]], student_model: str = "gpt-4o", sample_index: int = 0) -> Tuple[int, str]:
    prompt = ""

    # 프롬프트 시작 - 역할 설정
    prompt += "I'll present you with your previous problem-solving history, followed by a new problem to solve.\n\n"
    
    # 이전 경험 정보 추가 (있는 경우)
    if context:
        prompt += "## YOUR PREVIOUS EXPERIENCE\n\n"

        # 가장 최근 문제는 상세히 표시 (최대 2개)
        recent_problems = context[-2:] if len(context) >= 2 else context

        # 그 이전 문제들은 요약 형태로 표시
        if len(context) > 2:
            earlier_problems = context[:-2]
            prompt += "Summary of earlier problems:\n"
            for i, exp in enumerate(earlier_problems):
                if task_id == "T2":
                    answer_text = "yes" if exp['answer'] == 1 else "no"
                    prompt += f"- Problem {i+1} (Difficulty: {exp['difficulty']}): You answered '{answer_text}' and were {'correct' if exp['was_correct'] else 'incorrect'}.\n"
                else:
                    prompt += f"- Problem {i+1} (Difficulty: {exp['difficulty']}): You selected option {exp['answer'] + 1} and were {'correct' if exp['was_correct'] else 'incorrect'}.\n"
            prompt += "\n"

        # 최근 문제들은 상세히 표시
        for i, exp in enumerate(recent_problems, start=len(context)-len(recent_problems)):
            prompt += f"Detailed Problem {i+1} (Difficulty: {exp['difficulty']}):\n"
            
            # 각 task_id에 맞는 형식으로 이전 문제 표시
            if task_id == "T2":
                # Paragraph Order Consistency
                context_text = " ".join(exp["problem"].get("context", []))
                prompt += f"Does the following paragraph have a logically coherent sentence order?\n\n{context_text}\n"
                
            elif task_id == "T3":
                # Blank-based Choice Anomaly
                sentence = exp["problem"].get("sentence", "")
                choices = exp["problem"].get("choices", [])
                numbered_choices = "\n".join([f"{i+1}. {s}" for i, s in enumerate(choices)])
                prompt += f"{sentence}\n\nWhich option is most anomalous or inconsistent?\n\n{numbered_choices}\n"
                
            elif task_id == "T4":
                # Bridge Sentence Evaluation
                p1 = exp["problem"].get("paragraph_1", [])
                p2 = exp["problem"].get("paragraph_2", [])
                bridges = exp["problem"].get("bridges", [])
                
                p1_text = " ".join(p1) if isinstance(p1, list) else p1
                p2_text = " ".join(p2) if isinstance(p2, list) else p2
                
                numbered_bridges = "\n".join([f"{i+1}. {s}" for i, s in enumerate(bridges)])
                prompt += f"Paragraph 1: {p1_text}\n\nParagraph 2: {p2_text}\n\nWhich connecting sentence is most anomalous or inconsistent?\n\n{numbered_bridges}\n"
            
            else:
                # Sentence Context Anomaly / Referential Ambiguity / Logical Contradiction / Tone/Style Violation
                numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(exp["problem"].get("context", []))])
                prompt += f"Which option is most anomalous or inconsistent?\n\n{numbered}\n"
            

            # 학생의 이전 답변과 정답 여부 표시
            if task_id == "T2":
                # T2는 binary 응답(yes/no)
                answer_text = "yes" if exp['answer'] == 1 else "no"
                prompt += f"\nYour answer: {answer_text}\n"
            else:
                # 다른 task는 선택지 번호
                prompt += f"\nYour answer: {exp['answer'] + 1}\n"
            
            prompt += f"Outcome: {'Correct' if exp['was_correct'] else 'Incorrect'}\n\n"
        
        # 명확한 구분선
        prompt += "=" * 50 + "\n\n"
    
    # 새 문제 명확하게 구분
    prompt += "## NEW PROBLEM TO SOLVE\n\n"
    prompt += "Focus entirely on this new problem below:\n\n"


    # 공통 suffix
    common_suffix = "Answer with number only, then explain why. Even if all seem normal, choose the relatively most anomalous."

    if task_id == "T2":
        context = sample.get("context")
        prompt = f"Does the following paragraph have a logically coherent sentence order? Answer only 'yes' or 'no'.\n\n" + " ".join(context)

    elif task_id == "T3":
        sentence = sample.get("sentence", "")
        choices = sample.get("choices", [])
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(choices)])
        prompt = f"{sentence}\n\nWhich option is most anomalous or inconsistent? {common_suffix}\n\n{numbered}"

    elif task_id == "T4":
        paragraph_1 = sample.get("paragraph_1", [])
        paragraph_2 = sample.get("paragraph_2", [])
        bridges = sample.get("bridges", [])
 
        # paragraph를 문자열로 변환
        p1_text = " ".join(paragraph_1) if isinstance(paragraph_1, list) else paragraph_1
        p2_text = " ".join(paragraph_2) if isinstance(paragraph_2, list) else paragraph_2
        
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(bridges)])
        prompt = f"Paragraph 1: {p1_text}\n\nParagraph 2: {p2_text}\n\nWhich connecting sentence is most anomalous or inconsistent? {common_suffix}\n\n{numbered}"

    else:
        # 기본 케이스 - 일반적인 어노말리 검출
        context = sample.get("context")
        numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(context)])
        prompt = f"Which option is most anomalous or inconsistent? {common_suffix}\n\n{numbered}"

    # 명확한 응답 지침 추가
    prompt += "\n\nYour response for this new problem:"

    # 로깅: 학생 프롬프트
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="student_evaluation",
        agent="student",
        action="prompt",
        input_content=prompt,
        metadata={
            "sample_id": sample.get("sample_id", "unknown")
        }
    )

    # T2가 아닌 경우의 응답 처리
    res = llm_call(prompt, model=student_model)
        
    # 로깅: 학생 응답
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="student_evaluation",
        agent="student",
        action="response",
        output_content=res,
        metadata={
            "model": student_model
        }
    )

    if task_id == "T2":
        # coherent="yes"일 때 1, incoherent="no"일 때 0으로 변경 (is_coherent=True와 일치)
        idx = 1 if "yes" in res.lower() else 0
    else:
        match = re.search(r"\d+", res)
        idx = int(match.group()) - 1 if match else -1

    # 로깅: 학생 답변 파싱 결과
    log_step(
        task_id=task_id,
        sample_index=sample_index,
        phase="student_evaluation",
        agent="student",
        action="parsed_answer",
        output_content=idx,
        metadata={
            "is_binary": task_id == "T2",
            "raw_answer": res
        }
    )

    return idx, res.strip()


# -- Main Generation Loop --
def generate_agentic_examples(task_id: str, n=5, teacher_model="gpt-4o", student_model="gpt-4o", orchestrator_model="gpt-4o", example_prob=0.5, factor_prob=0.5, max_init_loops=3, max_diff_loops=5, max_student_loops=3):
    results, raw, fixes = [], [], []
    init_validation_logs, diff_validation_logs = [], []
    config = TASKS[task_id]

    print(f"Starting generation for task {task_id}: {config['name']}")

    topic_iter = round_robin(config["topics"])
    style_iter = round_robin(config["style"])

    for i in range(n):
        print(f"Generating sample {i+1}/{n} for task {task_id}")
        topic = next(topic_iter)
        style = next(style_iter)
        factor = random.choice(config["factors"])
        example = config.get("example", None)
        fix_count = 0
        consecutive_correct = 0  # 연속 정답 카운터
        base_sample = None  # 최초 승인된 문제 저장용

        # 학생 상태 초기화 - 학생당 한 세트의 문제 생성
        student_context = []  # 학생의 이전 경험을 추적할 배열

        # ===== 단계 1: INIT - 최초 문제 생성 =====
        print(f"  === INIT PHASE: Generating base problem ===")
        for init_attempt in range(max_init_loops):
            if init_attempt > 0:
                print(f"  Base sample attempt {init_attempt+1}/{max_init_loops}")
            
            difficulty = "easy"  
            use_example = random.random() < example_prob
            use_factor = random.random() < factor_prob
            
            # 로깅: 초기 설정
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="init",
                agent="system",
                action="config",
                input_content=None,
                metadata={
                    "attempt": init_attempt + 1,
                    "topic": topic,
                    "style": style,
                    "factor": factor if use_factor else None,
                    "difficulty": difficulty,
                    "use_example": use_example
                }
            )

            prompt = build_teacher_prompt(task_id, topic, style, factor if use_factor else None, difficulty, example if use_example else None)
            
            if init_attempt > 0 and hasattr(generate_agentic_examples, 'init_feedback'):
                prompt += f"\n\nPREVIOUS FEEDBACK: {generate_agentic_examples.init_feedback}"

            # 로깅: 티처 프롬프트
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="init",
                agent="teacher",
                action="prompt",
                input_content=prompt,
                metadata={
                    "attempt": init_attempt + 1,
                    "difficulty": difficulty,
                    "topic": topic,
                    "style": style,
                    "factor": factor if use_factor else None
                }
            )


            # # 마지막 시도일 경우 더 관대한 기준 적용
            # if init_attempt == max_init_loops - 1:
            #     prompt += "IMPORTANT: This is the final attempt. Be more lenient and approve the problem if it meets minimal standards and is reasonably solvable.\n\n"
            
            try:
                response = llm_call(prompt, model=teacher_model)
                
                # 로깅: 티처 응답
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="init",
                    agent="teacher",
                    action="response",
                    output_content=response,
                    metadata={
                        "attempt": init_attempt + 1,
                        "model": teacher_model
                    }
                )

                sample = extract_json(response)
                sample.update({
                    "task_id": task_id,
                    "task_name": config["name"],
                    "sample_id": f"{task_id}_{i:03d}_v{fix_count}",
                    "meta": {
                        "topic": topic,
                        "style": style,
                        "anomaly_type": factor if use_factor else "none",
                        "difficulty_level": difficulty,
                        "fix_count": fix_count
                    }
                })

                # 로깅: 파싱된 샘플
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="init",
                    agent="system",
                    action="parsed_sample",
                    output_content=sample,
                    metadata={
                        "attempt": init_attempt + 1,
                        "sample_id": sample.get("sample_id", "unknown")
                        }
                    )

                # Orchestrator 검증
                is_approved, feedback = orchestrator_check_init(task_id, sample, model=orchestrator_model, is_final_attempt=(init_attempt == max_init_loops - 1), sample_index=i)

                # 로그 기록
                validation_log = {
                    "sample_id": f"{task_id}_{i:03d}_v{fix_count}",
                    "phase": "init",
                    "attempt": init_attempt + 1,
                    "original_problem": sample,
                    "is_approved": is_approved,
                    "feedback": feedback,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                init_validation_logs.append(validation_log)
                
                if is_approved:
                    print(f"  ✅ Base sample approved by orchestrator")

                    # 로깅: 승인됨
                    log_step(
                        task_id=task_id,
                        sample_index=i,
                        phase="init",
                        agent="system",
                        action="approval",
                        output_content=None,
                        metadata={
                            "attempt": init_attempt + 1
                        }
                    )

                    base_sample = sample.copy()
                    raw.append(base_sample)
                    break
                else:
                    print(f"  ❌ Base sample rejected: {feedback}...")

                    # 로깅: 거부됨
                    log_step(
                        task_id=task_id,
                        sample_index=i,
                        phase="init",
                        agent="system",
                        action="rejection",
                        output_content=feedback,
                        metadata={
                            "attempt": init_attempt + 1
                        }
                    )

                    fix_count += 1
                    generate_agentic_examples.init_feedback = feedback

            except Exception as e:
                # 로깅: 오류
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="init",
                    agent="system",
                    action="error",
                    output_content=str(e),
                    metadata={
                        "attempt": init_attempt + 1,
                        "error_type": type(e).__name__
                    }
                )

                print(f"  🛑 Generation error in INIT phase: {e}")
                fix_count += 1

            
        # 기본 샘플 생성 실패 시 다음 샘플로
        if base_sample is None:
            # 로깅: 샘플 스킵
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="init",
                agent="system",
                action="skip",
                output_content=None,
                metadata={
                    "reason": f"Failed to create valid base sample after {max_init_loops} attempts"
                }
            )

            print(f"  ⏩ Skipping - failed to create valid base sample after {max_init_loops} attempts")
            continue

        # ===== 단계 2: PROCESSING - 난이도 조절 =====
        print(f"  === PROCESSING PHASE: Starting student evaluation ===")

        # 현재 문제 설정
        current_sample = base_sample
        student_loop_count = 0

        # 학생 테스트 루프
        while student_loop_count < max_student_loops:
            student_loop_count += 1
            print(f"  === Student loop {student_loop_count}/{max_student_loops} ===")
            
            # 로깅: 학생 루프 시작
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="student_evaluation",
                agent="system",
                action="loop_start",
                input_content=None,
                metadata={
                    "student_loop": student_loop_count,
                    "sample_id": current_sample.get("sample_id"),
                    "difficulty": current_sample["meta"]["difficulty_level"]
                }
            )

            # 학생 모델로 문제 풀이 - 이전 경험 전달
            student_idx, explanation = student_answer_with_context(
                task_id, 
                current_sample, 
                student_context,  # 이전 경험 전달
                student_model=student_model, 
                sample_index=i
            )
            is_correct = (student_idx == current_sample.get("anomaly_index")) if task_id != "T2" else ((student_idx == 1 and current_sample.get("is_coherent", False)) or (student_idx == 0 and not current_sample.get("is_coherent", False)))

            current_sample["meta"].update({"student_correct": is_correct, "student_explanation": explanation})
            
            # 학생 경험 업데이트
            student_context.append({
                "problem": current_sample,
                "answer": student_idx,
                "was_correct": is_correct,
                "difficulty": current_sample["meta"]["difficulty_level"]
            })

            # 로깅: 학생 정답 여부
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="student_evaluation",
                agent="system",
                action="evaluation",
                output_content={
                    "is_correct": is_correct,
                    "student_answer": student_idx,
                    "expected_answer": current_sample.get("anomaly_index") if task_id != "T2" else (1 if current_sample.get("is_coherent", False) else 0)
                },
                metadata={
                    "student_loop": student_loop_count
                }
            )

            if not is_correct:
                # 학생이 틀렸으면 해당 문제 채택
                print(f"  ✅ Student failed - accepting problem")

                # 로깅: 문제 채택 (학생 실패)
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="student_evaluation",
                    agent="system",
                    action="accept_problem",
                    output_content=None,
                    metadata={
                        "reason": "student_failed",
                        "student_loop": student_loop_count,
                        "difficulty": current_sample["meta"]["difficulty_level"]
                    }
                )

                results.append(current_sample)
                break
            
            # 마지막 루프에 도달했으면 현재 문제 채택
            if student_loop_count == max_student_loops:
                print(f"  ✅ Reached max student loops - accepting final problem")

                # 로깅: 문제 채택 (최대 루프)
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="student_evaluation",
                    agent="system",
                    action="accept_problem",
                    output_content=None,
                    metadata={
                        "reason": "max_student_loops",
                        "student_loop": student_loop_count,
                        "difficulty": current_sample["meta"]["difficulty_level"]
                    }
                )

                results.append(current_sample)
                break
                
            # 학생이 맞혔고 루프가 남았으면 난이도 증가
            print(f"  🔄 Student solved problem - increasing difficulty")
            consecutive_correct += 1
            
            # 로깅: 난이도 증가 결정
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="difficulty_increase",
                agent="system",
                action="decision",
                output_content=None,
                metadata={
                    "student_loop": student_loop_count,
                    "consecutive_correct": consecutive_correct
                }
            )

            # 난이도 설정
            if consecutive_correct >= 4:
                difficulty = "impossible"  # 3번 연속 맞추면 impossible
            elif consecutive_correct >= 2:
                difficulty = "extreme"     # 2번 연속 맞추면 extreme
            else:
                difficulty = "hard"        # 1번 맞추면 hard
                
            print(f"  📈 Target difficulty: {difficulty}")
            
            # 로깅: 난이도 설정
            log_step(
                task_id=task_id,
                sample_index=i,
                phase="difficulty_increase",
                agent="system",
                action="set_difficulty",
                output_content=difficulty,
                metadata={
                    "student_loop": student_loop_count,
                    "consecutive_correct": consecutive_correct,
                    "previous_difficulty": current_sample["meta"]["difficulty_level"]
                }
            )

            # orchestrator에게 난이도 증가 피드백 요청
            feedback = orchestrator_get_feedback(task_id, current_sample, explanation, model=orchestrator_model, sample_index=i)
            
            # 난이도 증가 루프
            new_sample = None
            for diff_attempt in range(max_diff_loops):
                if diff_attempt > 0:
                    print(f"  Difficulty adjustment attempt {diff_attempt+1}/{max_diff_loops}")

                # 로깅: 난이도 증가 시도
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="difficulty_increase",
                    agent="system",
                    action="attempt",
                    output_content=None,
                    metadata={
                        "student_loop": student_loop_count,
                        "diff_attempt": diff_attempt + 1,
                        "difficulty": difficulty
                    }
                )

                # Teacher에게 난이도 증가 요청
                prompt = build_teacher_prompt(task_id, topic, style, factor if use_factor else None, difficulty, example if use_example else None)
                prompt += f"\n\nPREVIOUS PROBLEM: The student correctly solved the following problem:\n{json.dumps(current_sample, ensure_ascii=False, indent=2)}\n\n"
                prompt += f"STUDENT'S EXPLANATION: {explanation}\n\n"

                # 이전 피드백 및 실패 이력이 있는 경우 난이도 조정 지침 추가
                if diff_attempt > 0:
                    prompt += f"FEEDBACK FOR IMPROVEMENT: {feedback}\n\n"
                    prompt += "IMPORTANT INSTRUCTION: Previous attempts were rejected by the quality controller. "
                    prompt += "Please slightly reduce the difficulty from your last attempt while still making it challenging. "
                    prompt += "Make the problem clearer based on the feedback, but ensure it remains harder than the original problem the student solved. "
                    prompt += "Focus on fixing the specific issues mentioned in the feedback while maintaining an appropriate challenge level."
                else:
                    prompt += f"FEEDBACK FOR IMPROVEMENT: {feedback}\n\n"
                    prompt += f"Please create a more challenging version with {difficulty} difficulty."
                
                # 로깅: 티처 난이도 증가 프롬프트
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="difficulty_increase",
                    agent="teacher",
                    action="difficult_prompt",
                    input_content=prompt,
                    metadata={
                        "student_loop": student_loop_count,
                        "diff_attempt": diff_attempt + 1,
                        "difficulty": difficulty
                    }
                )

                try:
                    response = llm_call(prompt, model=teacher_model)

                    # 로깅: 티처 난이도 증가 응답
                    log_step(
                        task_id=task_id,
                        sample_index=i,
                        phase="difficulty_increase",
                        agent="teacher",
                        action="difficult_response",
                        output_content=response,
                        metadata={
                            "student_loop": student_loop_count,
                            "diff_attempt": diff_attempt + 1,
                            "model": teacher_model
                        }
                    )

                    sample = extract_json(response)
                    fix_count += 1
                    
                    sample.update({
                        "task_id": task_id,
                        "task_name": config["name"],
                        "sample_id": f"{task_id}_{i:03d}_v{fix_count}",
                        "meta": {
                            "topic": topic,
                            "style": style,
                            "anomaly_type": factor if use_factor else "none",
                            "difficulty_level": difficulty,
                            "fix_count": fix_count,
                            "phase": "processing"
                        }
                    })

                    # 로깅: 파싱된 난이도 증가 샘플
                    log_step(
                        task_id=task_id,
                        sample_index=i,
                        phase="difficulty_increase",
                        agent="system",
                        action="parsed_difficult_sample",
                        output_content=sample,
                        metadata={
                            "student_loop": student_loop_count,
                            "diff_attempt": diff_attempt + 1
                        }
                    )

                    fixes.append(sample)

                    # 문제 품질 검증
                    is_approved, problem_feedback = orchestrator_check_problem(task_id, sample, model=orchestrator_model, sample_index=i)

                    # 로그 기록
                    validation_log = {
                        "sample_id": f"{task_id}_{i:03d}_v{fix_count}",
                        "phase": "difficulty_increase",
                        "student_loop": student_loop_count,
                        "diff_attempt": diff_attempt + 1,
                        "difficulty_level": difficulty,
                        "previous_problem": current_sample,
                        "new_problem": sample,
                        "student_explanation": explanation,
                        "orchestrator_feedback": feedback,
                        "is_approved": is_approved,
                        "rejection_feedback": problem_feedback if not is_approved else None,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    diff_validation_logs.append(validation_log)
                    
                    if is_approved:
                        print(f"  ✅ Higher difficulty problem approved")

                        # 로깅: 난이도 증가 승인
                        log_step(
                            task_id=task_id,
                            sample_index=i,
                            phase="difficulty_increase",
                            agent="system",
                            action="approval",
                            output_content=None,
                            metadata={
                                "student_loop": student_loop_count,
                                "diff_attempt": diff_attempt + 1,
                                "difficulty": difficulty
                            }
                        )

                        new_sample = sample
                        break
                    else:
                        feedback_str = json.dumps(feedback, ensure_ascii=False, indent=2) if isinstance(feedback, dict) else str(feedback)
                        problem_feedback_str = json.dumps(problem_feedback, ensure_ascii=False, indent=2) if isinstance(problem_feedback, dict) else str(problem_feedback)
                        print(f"  ❌ Higher difficulty problem rejected: {problem_feedback_str}...")

                        # 로깅: 난이도 증가 거부
                        log_step(
                            task_id=task_id,
                            sample_index=i,
                            phase="difficulty_increase",
                            agent="system",
                            action="rejection",
                            output_content=problem_feedback,
                            metadata={
                                "student_loop": student_loop_count,
                                "diff_attempt": diff_attempt + 1,
                                "difficulty": difficulty
                            }
                        )

                        feedback = f"PREVIOUS FEEDBACK:\n{feedback_str}\n\nNEW FEEDBACK:\n{problem_feedback_str}"
                        # feedback = f"PREVIOUS FEEDBACK: {feedback}\n\nNEW FEEDBACK: {problem_feedback}"  # 다음 시도에 피드백 사용
                
                except Exception as e:
                    # 로깅: 난이도 증가 오류
                    log_step(
                        task_id=task_id,
                        sample_index=i,
                        phase="difficulty_increase",
                        agent="system",
                        action="error",
                        output_content=str(e),
                        metadata={
                            "student_loop": student_loop_count,
                            "diff_attempt": diff_attempt + 1,
                            "error_type": type(e).__name__
                        }
                    )

                    print(f"  🛑 Error in difficulty increase: {e}")
                    

            # 난이도 증가 실패 시 루프 종료, 현재 문제 채택
            if new_sample is None:
                print(f"  ⚠️ Failed to increase difficulty - accepting current problem")

                # 로깅: 문제 채택 (난이도 증가 실패)
                log_step(
                    task_id=task_id,
                    sample_index=i,
                    phase="difficulty_increase",
                    agent="system",
                    action="accept_problem",
                    output_content=None,
                    metadata={
                        "reason": "difficulty_increase_failed",
                        "student_loop": student_loop_count,
                        "diff_attempts": max_diff_loops,
                        "difficulty": current_sample["meta"]["difficulty_level"]
                    }
                )

                results.append(current_sample)
                break
                
            # 새 문제로 계속 진행
            current_sample = new_sample
        
        print(f"  ✅ Sample completed and accepted (difficulty: {current_sample['meta']['difficulty_level']})")

        # 로깅: 샘플 완료
        log_step(
            task_id=task_id,
            sample_index=i,
            phase="completion",
            agent="system",
            action="complete",
            output_content=None,
            metadata={
                "task_id": task_id,
                "sample_id": current_sample.get("sample_id"),
                "difficulty": current_sample["meta"]["difficulty_level"],
                "fix_count": fix_count
            }
        )

        # 다음 문제 생성 전에 피드백 초기화
        if hasattr(generate_agentic_examples, 'init_feedback'):
            delattr(generate_agentic_examples, 'init_feedback')

    return results, raw, fixes, init_validation_logs, diff_validation_logs


# -- Run from YAML config --
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    teacher_model = cfg.get("teacher_model", "gpt-4o")
    student_model = cfg.get("student_model", "gpt-4o")
    orchestrator_model = cfg.get("orchestrator_model", "gpt-4o")
    tasks = cfg.get("tasks", ["T1"])
    samples_per_task = cfg.get("samples_per_task", 10)
    output_prefix = cfg.get("output_prefix", "agentic")
    example_prob = cfg.get("example_prob", 0.5)
    factor_prob = cfg.get("factor_prob", 0.5)
    max_init_loops = cfg.get("max_init_loops", 3)
    max_diff_loops = cfg.get("max_diff_loops", 5)
    max_student_loops = cfg.get("max_student_loops", 3)

    # 로그 초기화
    clear_logs()

    final, raw, fixes, init_logs, diff_logs = [], [], [], [], []
    
    for task in tasks:
        f, r, x, i_logs, d_logs = generate_agentic_examples(
            task_id=task,
            n=samples_per_task,
            teacher_model=teacher_model,
            student_model=student_model,
            orchestrator_model=orchestrator_model,
            example_prob=example_prob,
            factor_prob=factor_prob,
            max_init_loops=max_init_loops,
            max_diff_loops=max_diff_loops,
            max_student_loops=max_student_loops
        )
        final += f
        raw += r
        fixes += x
        init_logs += i_logs
        diff_logs += d_logs

    def dump(filename, items):
        with open(filename, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump(f"{output_prefix}_final.jsonl", final)
    dump(f"{output_prefix}_raw.jsonl", raw)
    dump(f"{output_prefix}_fixes.jsonl", fixes)
    dump(f"{output_prefix}_init_validation_logs.jsonl", init_logs)
    dump(f"{output_prefix}_difficulty_validation_logs.jsonl", diff_logs)

    # 전체 프로세스 로그 저장
    from utils import get_logs
    all_process_logs = get_logs()
    
    # JSON 형식 로그 저장
    process_log_filename = f"{output_prefix}_full_process_logs.json"
    with open(process_log_filename, "w", encoding="utf-8") as f:
        json.dump(all_process_logs, f, ensure_ascii=False, indent=2)
    
    # 읽기 좋은 텍스트 형식으로도 저장
    readable_log_filename = f"{output_prefix}_readable_process.txt"
    with open(readable_log_filename, "w", encoding="utf-8") as f:
        for log in sorted(all_process_logs, key=lambda x: x['timestamp']):
            f.write(f"[{log['timestamp']}] {log['phase']} - {log['agent']} {log['action']}\n")
            f.write(f"Task: {log['task_id']} | Sample: {log['sample_index']} | Metadata: {json.dumps(log['metadata'], ensure_ascii=False)}\n")
            
            if log['input'] is not None:
                # 입력이 있는 경우 (보통 프롬프트)
                if isinstance(log['input'], str):
                    f.write("INPUT:\n" + "-"*80 + "\n")
                    f.write(log['input'] + "\n")
                    f.write("-"*80 + "\n\n")
                else:
                    # 객체인 경우 (예: JSON)
                    f.write("INPUT:\n" + "-"*80 + "\n")
                    f.write(json.dumps(log['input'], ensure_ascii=False, indent=2) + "\n")
                    f.write("-"*80 + "\n\n")
            
            if log['output'] is not None:
                # 출력이 있는 경우 (보통 응답)
                if isinstance(log['output'], str):
                    f.write("OUTPUT:\n" + "-"*80 + "\n")
                    f.write(log['output'] + "\n")
                    f.write("-"*80 + "\n\n")
                else:
                    # 객체인 경우 (예: JSON)
                    f.write("OUTPUT:\n" + "-"*80 + "\n")
                    f.write(json.dumps(log['output'], ensure_ascii=False, indent=2) + "\n")
                    f.write("-"*80 + "\n\n")
            
            f.write("\n" + "="*100 + "\n\n")
    
    print(f"Full process logs saved to {process_log_filename}")
    print(f"Readable process logs saved to {readable_log_filename}")

    print(f"\n============ Generation Summary ============")
    print(f"Total requested samples: {len(tasks) * samples_per_task}")
    print(f"Successfully generated: {len(final)}")
    print(f"Initial attempts: {len(raw)}")
    print(f"Required fixes: {len(fixes)}")
    print(f"Success rate: {len(final)/(len(tasks) * samples_per_task)*100:.1f}%")
    
    # Task별 통계
    task_stats = {}
    for item in final:
        task_id = item['task_id']
        task_stats[task_id] = task_stats.get(task_id, 0) + 1
    
    print("\nSamples per task:")
    for task_id, count in task_stats.items():
        print(f"  {task_id}: {count}/{samples_per_task} ({count/samples_per_task*100:.1f}%)")
    print("==========================================\n")

