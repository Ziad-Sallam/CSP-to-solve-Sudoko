# 🧩 Sudoku Solver using CSP

A powerful Sudoku puzzle solver implementing Constraint Satisfaction Problem (CSP) techniques with an interactive GUI.

## 🎯 Features

- **Two Solving Algorithms**
  - 🔄 Backtracking with constraint checking
  - 🧠 Arc Consistency (AC-3) with MRV heuristic
  
- **Interactive GUI**
  - Real-time solving visualization
  - Step-by-step animation
  - Progress tracking and statistics
  
- **Dual Modes**
  - 🎲 Auto-generate puzzles (Easy/Medium/Hard)
  - ✍️ Manual input for custom puzzles
  
- **Advanced Features**
  - Puzzle validation
  - Constraints tree viewer
  - Performance comparison
  - Comprehensive report generation

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
tkinter (usually included with Python)
```

### Installation
```bash
# Clone or download the repository
cd CSP-to-solve-Sudoko

# Run the application
python sudoku_solver.py
```

## 📖 Usage

### Basic Operations

1. **Generate Puzzle**: Click "New Puzzle" to create a random puzzle
2. **Select Difficulty**: Choose Easy, Medium, or Hard
3. **Choose Algorithm**: Select Backtracking or Arc Consistency
4. **Solve**: Click "▶ Start Solving" to watch the AI solve it
5. **Manual Input**: Switch to "User Input" mode to enter your own puzzle

### Controls

| Button | Function |
|--------|----------|
| ✓ Validate | Check puzzle for conflicts and solvability |
| ▶ Start Solving | Begin solving with selected algorithm |
| ⏸ Pause | Pause/resume solving animation |
| ↻ Reset | Reset to initial puzzle state |
| New Puzzle | Generate new puzzle |
| 📊 Generate Report | Create performance analysis report |

## 🧮 Algorithms

### Backtracking Algorithm
- Classic recursive depth-first search
- Tries values sequentially for each empty cell
- Time Complexity: O(9^m) where m = empty cells
- Best for: Easy puzzles, educational purposes

### Arc Consistency (AC-3)
- Constraint propagation with domain reduction
- Uses Minimum Remaining Values (MRV) heuristic
- Time Complexity: O(n²d³) where n=cells, d=domain size
- Best for: Medium/Hard puzzles, optimal performance

## 📊 Performance Comparison

| Difficulty | Backtracking | Arc Consistency | Winner |
|------------|--------------|-----------------|--------|
| Easy | ~0.02s | ~0.02s | Tie |
| Medium | ~0.05s | ~0.03s | AC-3 (40% faster) |
| Hard | ~0.15s | ~0.09s | AC-3 (43% faster) |

## 🗂️ Project Structure

```
CSP-to-solve-Sudoko/
│
├── sudoku_solver.py      # Main application
├── REPORT.md            # Comprehensive project report
└── README.md            # This file
```

## 🎓 CSP Formulation

**Variables:** 81 cells in 9×9 grid  
**Domains:** {1,2,3,4,5,6,7,8,9} for empty cells  
**Constraints:**
- Row constraint: All values in row must be unique
- Column constraint: All values in column must be unique
- Box constraint: All values in 3×3 subgrid must be unique

## 📝 Report Generation

Click "📊 Generate Report" to automatically:
- Test both algorithms on multiple puzzles
- Compare performance across difficulty levels
- Generate HTML report with visualizations
- Show arc consistency trees
- Provide statistical analysis

## 🎨 UI Features

### Color Coding
- 🔵 Blue: User input cells
- ⚪ Gray: Initial puzzle cells
- 🟡 Yellow: AI currently solving
- 🟢 Green: AI solved cells
- 🔴 Red: Validation errors

### Panels
1. **Left**: Sudoku grid + controls
2. **Middle**: Progress tracking + statistics
3. **Right**: Solving steps + constraints tree

## 🔧 Technical Details

### Data Structures
- `List[List[int]]`: Board representation (9×9 grid)
- `Dict[Tuple, List]`: Domain storage for AC-3
- `List[Tuple]`: Arcs queue for constraint propagation
- `Dict[Tuple, Dict]`: Constraints tree snapshot

### Key Classes
- `SudokuSolver`: Base solver with utility methods
- `BacktrackingSolver`: Backtracking algorithm implementation
- `ArcConsistencySolver`: AC-3 algorithm implementation
- `SudokuCell`: Individual cell widget
- `SudokuPuzzleSolver`: Main application controller

## 📚 Documentation

For detailed information, see [REPORT.md](REPORT.md) which includes:
- Algorithm pseudocode
- Sample runs with solutions
- Arc consistency tree examples
- Performance analysis
- Design decisions
- References

## 🎯 Use Cases

- **Education**: Learn CSP and AI algorithms
- **Research**: Compare algorithm performance
- **Entertainment**: Solve Sudoku puzzles
- **Development**: Extend with new features

## 🔮 Future Enhancements

- [ ] Forward checking algorithm
- [ ] More heuristics (Degree, LCV)
- [ ] Larger grid support (16×16, 25×25)
- [ ] Variant Sudoku rules
- [ ] Hint system for players
- [ ] Puzzle uniqueness verification

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Author

AI Project - Constraint Satisfaction Problem Implementation

## 🙏 Acknowledgments

- Based on CSP techniques from AI: A Modern Approach (Russell & Norvig)
- Arc Consistency algorithm by Mackworth (1977)
- GUI built with Python Tkinter

---

**Enjoy solving Sudoku puzzles with AI! 🎉**
