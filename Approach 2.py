# ============================================================
# KAGGLE SUBMISSION NOTEBOOK - ARC PRIZE 2025
# ============================================================

import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load test data
with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_challenges = json.load(f)

print(f"Loaded {len(test_challenges)} test tasks")

# ============================================================
# PATTERN MATCHING STRATEGIES
# ============================================================

def try_color_mapping(train_examples, test_input):
    try:
        if len(train_examples) == 0:
            return None
        ex = train_examples[0]
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape != out.shape:
            return None
        
        color_map = {}
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                color_map[inp[i,j]] = out[i,j]
        
        for ex in train_examples[1:]:
            inp2, out2 = np.array(ex['input']), np.array(ex['output'])
            if inp2.shape != out2.shape:
                return None
            for i in range(inp2.shape[0]):
                for j in range(inp2.shape[1]):
                    if color_map.get(inp2[i,j]) != out2[i,j]:
                        return None
        
        test_arr = np.array(test_input)
        result = np.zeros_like(test_arr)
        for i in range(test_arr.shape[0]):
            for j in range(test_arr.shape[1]):
                result[i,j] = color_map.get(test_arr[i,j], test_arr[i,j])
        return result.tolist()
    except:
        return None

def try_tiling(train_examples, test_input):
    try:
        ex = train_examples[0]
        inp, out = np.array(ex['input']), np.array(ex['output'])
        h_ratio = out.shape[0] // inp.shape[0]
        w_ratio = out.shape[1] // inp.shape[1]
        if h_ratio < 1 or w_ratio < 1:
            return None
        
        expected = np.tile(inp, (h_ratio, w_ratio))
        if expected.shape != out.shape or not np.array_equal(expected, out):
            return None
        
        for ex in train_examples[1:]:
            inp2, out2 = np.array(ex['input']), np.array(ex['output'])
            expected2 = np.tile(inp2, (h_ratio, w_ratio))
            if expected2.shape != out2.shape or not np.array_equal(expected2, out2):
                return None
        
        test_arr = np.array(test_input)
        result = np.tile(test_arr, (h_ratio, w_ratio))
        return result.tolist()
    except:
        return None

def try_rotation_flip(train_examples, test_input):
    try:
        ex = train_examples[0]
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape != out.shape:
            return None
        
        # Try rotations
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, k), out):
                valid = all(np.array_equal(np.rot90(np.array(ex2['input']), k), 
                           np.array(ex2['output'])) for ex2 in train_examples[1:])
                if valid:
                    return np.rot90(np.array(test_input), k).tolist()
        
        # Try flips
        for axis in [0, 1]:
            if np.array_equal(np.flip(inp, axis), out):
                valid = all(np.array_equal(np.flip(np.array(ex2['input']), axis),
                           np.array(ex2['output'])) for ex2 in train_examples[1:])
                if valid:
                    return np.flip(np.array(test_input), axis).tolist()
        return None
    except:
        return None

def pattern_matching_solver(task):
    train_examples = task['train']
    test_input = task['test'][0]['input']
    candidates = []
    
    c = try_color_mapping(train_examples, test_input)
    if c: candidates.append(c)
    
    c = try_tiling(train_examples, test_input)
    if c: candidates.append(c)
    
    c = try_rotation_flip(train_examples, test_input)
    if c: candidates.append(c)
    
    return candidates

# ============================================================
# SIMPLE SOLVER (fallback)
# ============================================================

def simple_solve(task):
    """Fallback: just return input or try simple transformations"""
    test_input = task['test'][0]['input']
    train_out = task['train'][0]['output']
    
    # Try pattern matching first
    candidates = pattern_matching_solver(task)
    
    # Fallback options
    if len(candidates) < 2:
        candidates.append(test_input)  # Copy input
    if len(candidates) < 2:
        candidates.append(train_out)  # Copy first training output
    
    return candidates[:2]

# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = {}

for idx, (task_id, task) in enumerate(test_challenges.items(), 1):
    if idx % 10 == 0:
        print(f"Processing {idx}/{len(test_challenges)}...")
    
    try:
        attempts = simple_solve(task)
        task_outputs = []
        
        for test_item in task['test']:
            task_outputs.append({
                'attempt_1': attempts[0],
                'attempt_2': attempts[1]
            })
        
        submission[task_id] = task_outputs
        
    except Exception as e:
        print(f"Error on {task_id}: {e}")
        fallback = task['test'][0]['input']
        task_outputs = []
        for test_item in task['test']:
            task_outputs.append({
                'attempt_1': fallback,
                'attempt_2': fallback
            })
        submission[task_id] = task_outputs

# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"\nSubmission created for {len(submission)} tasks!")
print("File saved as submission.json")
