import json


def build_teacher_prompt(task_id, topic, style, factor, difficulty_level, example=None):
    
    prompt = f"You are a {style}-style exam question generator. Create a question for task {task_id} on the topic of {topic}.\n"

    # Difficulty-based instruction
    if difficulty_level == "impossible":
        difficulty_desc = "Create an exceptionally subtle anomaly that requires deep expertise to detect. The anomaly should be a nuanced conceptual inconsistency - not a factual error - seamlessly embedded within the text. It should challenge even expert readers while remaining logically defensible."
        # difficulty_desc = "Create an extraordinarily subtle and sophisticated anomaly that would challenge even the most expert critics. The anomaly should be so deeply integrated that it requires specialized knowledge and meticulous analysis to detect, yet remains logically defensible upon discovery. Design a problem at the absolute frontier of difficulty."
    elif difficulty_level == "extreme":
        difficulty_desc = "Make the anomaly exceptionally subtle and sophisticated. It should require deep critical thinking and expert knowledge to detect. Create a challenging problem that would be difficult even for advanced students."
    elif difficulty_level == "hard":
        difficulty_desc = "Make the anomaly logically plausible and contextually realistic, yet semantically or pragmatically inconsistent. Avoid clear contradictions."
    elif difficulty_level == "easy":
        difficulty_desc = "Use a clear but non-trivial anomaly. It should be detectable without requiring extensive critical analysis."
    else:  # medium (default)
        difficulty_desc = "Create a non-trivial anomaly that requires careful reading to detect. It should be noticeable but not immediately obvious."
        

    # Task-specific instructions
    if task_id == "T1":
        prompt += f"\nGenerate 5 to 6 sentences on {topic}. One of them should be anomalous (e.g., semantically inconsistent or conceptually off-topic)."
        if factor:
            prompt += f" The anomaly should be based on: {factor}."
        prompt += f" {difficulty_desc}"
        
        if example:
            prompt += f"\n\nHere is an example format to follow:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += f"\n\nReturn the result strictly in JSON format:\n{{\n  \"context\": [\"...\"],\n  \"anomaly_index\": <integer>,\n  \"meta\": {{\n    \"source\": \"{style}\",\n    \"topic\": \"{topic}\",\n    \"anomaly_type\": \"{factor if factor else 'none'}\"\n  }}\n}}"

    elif task_id == "T2":  # Paragraph Order Consistency
        prompt += f"\nCreate 5 sentences about {topic} with the following requirements:\n"
        prompt += "Randomly choose ONE of the following options:\n\n"
        
        prompt += "OPTION 1: Write logically coherent sentences in the correct order (is_coherent: true)\n\n"
        
        prompt += "OPTION 2: Write sentences with subtly disrupted order (is_coherent: false):\n"
        prompt += "   - Move 1-2 sentences that disrupt temporal/causal flow\n"
        prompt += "   - Avoid obvious scrambling; maintain some local coherence\n"
        prompt += "   - The disruption should be detectable but not immediately obvious\n\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn ONLY ONE JSON with:\n{\n  \"context\": [list of 5 sentences],\n  \"is_coherent\": boolean (true if in logical order, false if shuffled)\n}"

    elif task_id == "T3":  # Blank-based Choice Anomaly
        prompt += f"\nDesign a sentence completion question on {topic}:\n"
        prompt += "1. Write one sentence with a meaningful blank (marked as ___)\n"
        prompt += "2. Provide 5 choices where one is subtly inappropriate\n"
        if factor:
            prompt += f"3. The inappropriate choice should involve: {factor}\n"
        prompt += f"4. {difficulty_desc}\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn JSON with:\n{\n  \"sentence\": \"sentence with ___\",\n  \"choices\": [5 options],\n  \"anomaly_index\": index of the anomalous choice\n}"

    elif task_id == "T4":  # Bridge Sentence Evaluation
        prompt += f"\nCreate a paragraph bridging task on {topic}:\n"
        prompt += "1. Write two short paragraphs (2-3 sentences each)\n"
        prompt += "2. Generate 5 candidate bridge sentences\n"
        prompt += "3. One bridge should be contextually or logically weak\n"
        if factor:
            prompt += f"4. The weak bridge should have: {factor}\n"
        prompt += f"5. {difficulty_desc}\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn JSON with:\n{\n  \"paragraph_1\": [sentences],\n  \"paragraph_2\": [sentences],\n  \"bridges\": [5 bridge options],\n  \"anomaly_index\": index of the weak bridge\n}"

    elif task_id == "T5":  # Referential Ambiguity
        prompt += f"\nGenerate a coreference resolution question on {topic}:\n"
        prompt += "1. Write 5 sentences with pronouns and referents\n"
        prompt += "2. One sentence must contain ambiguous pronouns where it's unclear which antecedent the pronoun refers to\n"
        prompt += "3. The ambiguous sentence should create one clear type of referential ambiguity by either:\n"
        prompt += "   a) Using a pronoun with multiple possible antecedents - where the pronoun could refer to two or more previously mentioned entities, OR\n"
        prompt += "   b) Using a pronoun with an unclear or implied but not explicitly mentioned antecedent\n"
        prompt += "4. Choose only ONE of these ambiguity types (a or b) for the sentence, and ensure it is subtle yet detectable\n"

        # 모든 난이도에 적용되는 추가 지침
        prompt += "5. Make the potential referents semantically similar or related, increasing the subtlety of the ambiguity\n"

        # 난이도별 특화 지침
        if difficulty_level == "hard" or difficulty_level == "extreme":
            prompt += "6. Ensure the ambiguity requires careful context analysis to detect\n"

        if difficulty_level == "extreme":
            prompt += "7. Make the ambiguity multi-layered, with at least 2-3 possible interpretations for the ambiguous pronoun\n"
            prompt += "8. Place the ambiguous sentence in a complex paragraph structure where careful analysis is required to detect the ambiguity\n"

        if factor:
            prompt += f"9. The ambiguity should involve: {factor}\n"
        prompt += f"10. {difficulty_desc}\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn JSON with:\n{\n  \"context\": [5 sentences],\n  \"anomaly_index\": index of the ambiguous sentence\n}"

    elif task_id == "T6":  # Logical Contradiction
        prompt += f"\nCreate a logical consistency question on {topic}:\n"
        prompt += "1. Generate 5 statements\n"
        prompt += "2. One should contain a contradiction or reversed logic\n"
        if factor:
            prompt += f"3. The contradiction should involve: {factor}\n"
        prompt += f"4. {difficulty_desc}\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn JSON with:\n{\n  \"context\": [5 statements],\n  \"anomaly_index\": index of the contradictory statement\n}"

    elif task_id == "T7":  # Tone / Style Violation
        prompt += f"\nDesign a stylistic coherence question on {topic}:\n"
        prompt += "1. Write 5 sentences with consistent formal/academic tone\n"
        prompt += "2. One sentence should subtly violate the established tone/style\n"
        if factor:
            prompt += f"3. The violation should involve: {factor}\n"
        prompt += f"4. {difficulty_desc}\n"
        
        if example:
            prompt += f"\nExample format:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```"
        
        prompt += "\n\nReturn JSON with:\n{\n  \"context\": [5 sentences],\n  \"anomaly_index\": index of the tone-violating sentence\n}"

    return prompt