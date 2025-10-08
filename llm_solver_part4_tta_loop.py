
ARC Prize LLM Solver - Part 4: The Test-Time Adaptation (TTA) Loop
======================================================================

This script is the final orchestrator. It brings together all previous components
to create a full, self-correcting solver.

1.  It uses the PromptEngine to create an initial prompt.
2.  It simulates calling an LLM to get a candidate program.
3.  It uses the ProgramVerifier to check the program.
4.  If verification fails, it generates feedback and re-prompts the LLM,
    asking it to correct its mistake. This is the "Test-Time Adaptation" loop.
"""

from typing import List, Dict, Any, Optional
import numpy as np

# In a real notebook, these classes would be in prior cells.
from llm_solver_part1_dsl_and_serializer import DSL
from llm_solver_part2_prompt_engine import PromptEngine
from llm_solver_part3_executor_and_verifier import ProgramVerifier, DSLExecutor

class TTA_Solver:
    """
    Manages the end-to-end process of solving an ARC task using a 
    Test-Time Adaptation (TTA) loop with a simulated LLM.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.prompt_engine = PromptEngine()
        self.verifier = ProgramVerifier()
        self.executor = DSLExecutor()
        print("TTA Solver Initialized.")

    def _simulate_llm_call(self, prompt: str, attempt_number: int) -> str:
        """
        A placeholder function to simulate a call to an LLM API.
        
        It's hardcoded to demonstrate the TTA loop:
        - On the first attempt, it returns an incorrect program.
        - On subsequent attempts, it returns the correct program.
        """
        print(f"\n--- Attempt {attempt_number}: Sending prompt to LLM (length: {len(prompt)} chars) ---")
        
        # This is the correct program for the sample task used in the examples.
        correct_program = '[("flip", {"axis": "h"}), ("recolor", {"color_map": {1:7, 2:8, 3:5, 4:6}})]'
        
        # This is an incorrect program that only performs the flip.
        incorrect_program = '[("flip", {"axis": "h"})]'

        if attempt_number == 1:
            print("LLM is thinking... and returns an initial (incorrect) program.")
            return incorrect_program
        else:
            print("LLM is thinking... and returns a corrected program based on feedback.")
            return correct_program

    def _generate_feedback(self, failed_program_str: str, train_examples: List[Dict]) -> str:
        """
        Generates descriptive feedback for the LLM if its program fails.
        """
        feedback = ["\n# FEEDBACK\nYour previous program failed verification."]
        try:
            program = ast.literal_eval(failed_program_str)
            input_grid = np.array(train_examples[0]['input'])
            expected_output = np.array(train_examples[0]['output'])
            
            predicted_output = self.executor.execute_program(input_grid, program)

            if predicted_output.shape != expected_output.shape:
                feedback.append(f"Your program produced a grid of shape {predicted_output.shape}, but the expected shape was {expected_output.shape}.")
            else:
                feedback.append("The output grid shape was correct, but the colors did not match the expected output.")
            
            feedback.append("Please analyze the examples again and provide a new program.")

        except Exception as e:
            feedback.append(f"Your program could not be executed. Error: {e}")

        return "\n".join(feedback)

    def solve(self, task: Dict) -> Optional[List[List[int]]]:
        """
        Solves a single ARC task using the TTA loop.
        """
        print(f"\n{ '='*20 }\nSolving New Task\n{ '='*20 }")
        train_examples = task['train']
        test_input_grid = np.array(task['test'][0]['input'])
        
        # Start with the initial prompt
        current_prompt = self.prompt_engine.create_prompt(train_examples)
        
        for i in range(self.max_retries):
            attempt = i + 1
            
            # 1. Get program from LLM
            program_str = self._simulate_llm_call(current_prompt, attempt)
            print(f"LLM Response (Program):\n```python\n{program_str}\n```")

            # 2. Verify the program
            is_correct = self.verifier.verify(program_str, train_examples)
            print(f"Verification Result: {'✅ SUCCESS' if is_correct else '❌ FAILURE'}")

            if is_correct:
                # 3a. If correct, execute on test grid and return
                print("\nProgram is correct! Executing on the test input.")
                final_program = ast.literal_eval(program_str)
                final_prediction = self.executor.execute_program(test_input_grid, final_program)
                return final_prediction.tolist()
            else:
                # 3b. If incorrect, generate feedback and update the prompt for the next loop
                if i < self.max_retries - 1:
                    feedback = self._generate_feedback(program_str, train_examples)
                    current_prompt += f"\n{program_str}\n```\n{feedback}"
                    print(f"Generated Feedback for LLM: {feedback}")
                
        print("\nMaximum retries reached. Failed to find a correct solution.")
        return None

# Example of how to run the full TTA loop
if __name__ == '__main__':
    # 1. Define the same sample ARC task
    sample_task = {
        "train": [
            {
                "input": [[1, 2, 0], [3, 4, 0]],
                "output": [[0, 8, 7], [0, 6, 5]]
            }
        ],
        "test": [
            {
                "input": [[9, 1, 0], [0, 2, 0]],
            }
        ]
    }

    # 2. Instantiate the main solver
    solver = TTA_Solver(max_retries=2)

    # 3. Run the solver on the task
    final_prediction = solver.solve(sample_task)

    # 4. Print the final result
    print(f"\n{ '='*20 }\nFinal Result\n{ '='*20 }")
    if final_prediction:
        print("Successfully found a solution and generated a prediction:")
        # Pretty print the grid
        for row in final_prediction:
            print(" ".join(map(str, row)))
    else:
        print("Could not solve the task.")

    # Expected output of this script will show:
    # - Attempt 1 failing verification.
    # - Feedback being generated.
    # - Attempt 2 succeeding verification.
    # - The final prediction being generated.

