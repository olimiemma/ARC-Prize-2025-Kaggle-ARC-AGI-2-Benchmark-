
ARC Prize LLM Solver - Part 1: DSL Definition & Grid Serializer
================================================================

This script contains the foundational components for an LLM-based ARC solver.

1.  GridSerializer: Converts ARC grids into a text format suitable for an LLM prompt.
2.  DSL (Domain-Specific Language): Defines the set of functions that the LLM can use
    to construct a solution program. This definition will be injected into the prompt.
"""

from typing import List, Dict, Tuple

class GridSerializer:
    """
    Handles the conversion of ARC grids to and from text format for LLM prompts.
    """
    @staticmethod
    def to_text(grid: List[List[int]]) -> str:
        """
        Serializes a grid into a simple, human-readable string format.
        
        Example: [[1, 2], [0, 0]] -> "1 2\n0 0"
        """
        if not grid:
            return ""
        return "\n".join(" ".join(map(str, row)) for row in grid)

class DSL:
    """
    Defines the comprehensive Domain-Specific Language (DSL) for ARC tasks.
    
    This class serves as documentation for the LLM. It outlines the available
    functions, their parameters, and what they do. The actual implementation
    of these functions will be in the executor module.
    
    The DSL is synthesized from the most powerful operations found in the
    project's existing solvers (primarily a_v4.py).
    """

    @staticmethod
    def get_dsl_definition_for_prompt() -> str:
        """
        Returns a string detailing the DSL for inclusion in the LLM prompt.
        """
        return """
You must express your solution as a program, which is a list of function calls.
Each function call is a tuple: (function_name, {parameter_name: value, ...}).

Available functions:
--------------------

### Geometric Operations

1.  `rotate(degrees: int)`
    - Rotates the grid clockwise. `degrees` must be 90, 180, or 270.
    - Example: `("rotate", {"degrees": 90})`

2.  `flip(axis: str)`
    - Flips the grid. `axis` must be 'h' (horizontal) or 'v' (vertical).
    - Example: `("flip", {"axis": "h"})`

3.  `transpose()`
    - Swaps the grid's rows and columns.
    - Example: `("transpose", {})`

### Scaling and Tiling

4.  `tile_scale(height_ratio: int, width_ratio: int)`
    - Scales the grid up by repeating each cell.
    - Example: `("tile_scale", {"height_ratio": 2, "width_ratio": 2})`

5.  `block_reduce(height_ratio: int, width_ratio: int)`
    - Scales the grid down by taking the most common color in blocks.
    - Example: `("block_reduce", {"height_ratio": 2, "width_ratio": 2})`

### Cropping and Padding

6.  `crop_to_content()`
    - Crops the grid to the bounding box of all non-background colors.
    - Example: `("crop_to_content", {})`

7.  `pad_to_size(height: int, width: int, fill_color: int)`
    - Pads or crops the grid to a target size, centering the content.
    - Example: `("pad_to_size", {"height": 10, "width": 10, "fill_color": 0})`

8.  `add_border(thickness: int, color: int)`
    - Adds a border of a specific thickness and color around the grid.
    - Example: `("add_border", {"thickness": 1, "color": 0})`

9.  `remove_border(thickness: int)`
    - Removes a border of a specific thickness from the grid.
    - Example: `("remove_border", {"thickness": 1})`

### Object-Centric Operations
(These operate on connected components of the same color, excluding the background)

10. `keep_largest_object(fill_color: int)`
    - Keeps only the largest object and replaces the rest of the grid with `fill_color`.
    - Example: `("keep_largest_object", {"fill_color": 0})`

11. `crop_to_largest_object()`
    - Crops the grid to the bounding box of the largest object.
    - Example: `("crop_to_largest_object", {})`

### Color Operations

12. `recolor(color_map: dict)`
    - Changes colors in the grid based on a provided mapping.
    - Example: `("recolor", {"color_map": {1: 2, 5: 0}})`

### Other Transformations

13. `translate(dy: int, dx: int, fill_color: int)`
    - Shifts the grid content by (dy, dx), filling empty space with `fill_color`.
    - Example: `("translate", {"dy": 1, "dx": -1, "fill_color": 0})`

14. `symmetrize(axis: str)`
    - Forces the grid to be symmetrical along an axis by copying one half over the other.
    - `axis` must be 'h' (copies top to bottom) or 'v' (copies left to right).
    - Example: `("symmetrize", {"axis": "h"})`
"""

# Example of how to use these classes
if __name__ == '__main__':
    # 1. Define a sample grid
    sample_grid = [
        [0, 1, 1, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0]
    ]

    # 2. Serialize the grid for an LLM prompt
    serialized_grid = GridSerializer.to_text(sample_grid)
    print("--- Serialized Grid for LLM Prompt ---")
    print(serialized_grid)
    print("\n" + "="*40 + "\n")

    # 3. Get the DSL definition to include in the prompt
    dsl_definition = DSL.get_dsl_definition_for_prompt()
    print("--- DSL Definition for LLM Prompt ---")
    print(dsl_definition)
