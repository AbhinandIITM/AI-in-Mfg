with open('agent/train_rl.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
text = re.sub(r'\[.*?\] Training Complete!', '[OK] Training Complete!', text)
text = re.sub(r'  .*? Model Policy:', '  - Model Policy:', text)
text = re.sub(r'  .*? Normalization Stats:', '  - Normalization Stats:', text)
text = re.sub(r'  .*? TensorBoard Logs:', '  - TensorBoard Logs:', text)

with open('agent/train_rl.py', 'w', encoding='utf-8') as f:
    f.write(text)
