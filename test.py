import json
import re

txt = """
Thought: The classification of the image is based on the highest confidence classification while noting the validation insights.
Action: Final Answer
Action Input: {"answer": {"class_name": "BLI", "confidence": "100%", "explanation": "The initial classification of Blue Light Imaging (BLI) was made with high confidence, indicating enhanced visualization capabilities, which is suitable for polyp detection."}}
"""

a_input = re.search(r'Action Input:\s*(\{.*\})', txt, re.DOTALL)
print(a_input)

input_val = json.loads(a_input.group(1))
print(input_val)