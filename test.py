import json
import re

txt = 'Action Input: {"answer": {"class_name": "Ta_trang", "confidence": "99.9998%", "explanation": "The image was classified as Duodenum (Ta_trang) with high confidence, and validation confirmed this result with an even higher confidence level."}}'

a_input = re.search(r'Action Input:\s*(\{.*\})', txt, re.DOTALL)
print(a_input)

input_val = json.loads(a_input.group(1))
print(input_val)