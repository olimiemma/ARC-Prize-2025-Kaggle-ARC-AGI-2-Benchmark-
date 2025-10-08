
ARC Prize LLM Solver - Part 2: The Prompt Engine
==================================================

This script defines the `PromptEngine`, which is responsible for constructing
the complete, detailed prompt that will be sent to the LLM. 

It uses the components from Part 1 to assemble a prompt containing:
1. A clear explanation of the goal.
2. The full definition of the Domain-Specific Language (DSL).
3. The training examples for a specific task, serialized into text.
4. A final instruction telling the LLM how to format its response.


from typing import List, Dict

# We will import from our previously created file.
# In a real Kaggle notebook, these classes would be in the same cell or a previous one.
from llm_solver_part1_dsl_and_serializer import GridSerializer, DSL

class PromptEngine:
    """
    Constructs high-quality prompts for solving ARC tasks with an LLM.
    """
    
    # A master template for the entire prompt.
    MASTER_PROMPT_TEMPLATE = """
# MISSION: Solve an Abstract Reasoning Corpus (ARC) task

You are an expert AI assistant specializing in logic puzzles. Your goal is to deduce a general transformation rule from a set of training examples. Each example consists of an 'input' grid and an 'output' grid.

You must express this rule as a program using a specific Domain-Specific Language (DSL). The program should be a sequence of function calls that, when applied to any input grid, produces the corresponding output grid.

# DOMAIN-SPECIFIC LANGUAGE (DSL) DEFINITION

{dsl_definition}

# TASK EXAMPLES

Here are the training examples for the current task. Analyze them to find the underlying transformation rule.

{formatted_examples}

# YOUR TASK

Based on the examples, determine the single program that transforms every input grid to its corresponding output grid. Your response MUST be only the Python code for the program, which is a list of tuples. Do not include any other text, explanations, or markdown formatting.

Example of a valid response format:
```python
[("flip", {"axis": "v"}), ("recolor", {"color_map": {1: 8, 2: 3}})]
```

Your program:
```python
"

    def __init__(self):
        """
        Initializes the PromptEngine with the DSL definition.
        """
        self.dsl_definition = DSL.get_dsl_definition_for_prompt()

    def create_prompt(self, train_examples: List[Dict]) -> str:
        """
        Creates the full prompt for a given ARC task.

        Args:
            train_examples: A list of training pairs, where each pair is a dict
                            with 'input' and 'output' grid keys.

        Returns:
            A string containing the complete prompt to be sent to the LLM.
        """
        formatted_examples = []
        for i, pair in enumerate(train_examples):
            input_grid_str = GridSerializer.to_text(pair['input'])
            output_grid_str = GridSerializer.to_text(pair['output'])
            
            example_str = (
                f"## Example {i+1}\n\n"
                f"### Input Grid:\n{input_grid_str}\n\n"
                f"### Output Grid:\n{output_grid_str}"
            )
            formatted_examples.append(example_str)
        
        # Assemble the final prompt using the master template
        final_prompt = self.MASTER_PROMPT_TEMPLATE.format(
            dsl_definition=self.dsl_definition,
            formatted_examples="\n\n---\n\n".join(formatted_examples)
        )
        
        return final_prompt

# Example of how to use this class
if __name__ == '__main__':
    # 1. Define a sample ARC task (e.g., a simple flip and recolor)
    sample_task = {
        "train": [
            {
                "input": [
                    [1, 2, 0],
                    [3, 4, 0]
                ],
                "output": [
                    [0, 8, 7],
                    [0, 6, 5]
                ]
            },
            {
                "input": [
                    [5, 6],
                    [7, 8]
                ],
                "output": [
                    [2, 1],
                    [4, 3]
                ]
            }
        ]
    }

    # 2. Instantiate the prompt engine
    prompt_engine = PromptEngine()

    # 3. Create the prompt for the sample task
    final_prompt_for_llm = prompt_engine.create_prompt(sample_task['train'])

    # 4. Print the result to show what would be sent to the LLM
    print("--- Complete Prompt to be Sent to LLM ---")
    print(final_prompt_for_llm)

    # You can see how the grids are serialized and embedded into the larger template,
    # ready for an LLM to process.

