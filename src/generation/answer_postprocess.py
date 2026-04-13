from __future__ import annotations
import re
def postprocess_answer(raw_answer: str) -> str:
    answer= re.sub(r"<think>.*?</think>", "", raw_answer,flags=re.DOTALL)
    answer =  answer.strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer