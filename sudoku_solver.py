import tkinter as tk
from tkinter import ttk, messagebox
import time
import random
from typing import List, Tuple, Optional, Callable, Dict, Set

# Constants
GRID_SIZE = 9
EMPTY_CELL = 0

# Colors
COLOR_BG = "#ffffff"
COLOR_INITIAL = "#f1f5f9"
COLOR_USER = "#ffffff"
COLOR_AI_SOLVING = "#fef3c7"
COLOR_AI_SOLVED = "#10b981"
COLOR_ERROR = "#fee2e2"
COLOR_SELECTED = "#dbeafe"
COLOR_HOVER = "#f8fafc"
COLOR_GRID_THICK = "#334155"
COLOR_GRID_LINE = "#cbd5e1"
COLOR_PRIMARY = "#0ea5e9"


class SudokuSolver:
    @staticmethod
    def is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
        for c in range(GRID_SIZE):
            if c != col and board[row][c] == num:
                return False
        for r in range(GRID_SIZE):
            if r != row and board[r][col] == num:
                return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r != row or c != col) and board[r][c] == num:
                    return False
        
        return True
    @staticmethod
    def get_domain(board: List[List[int]], row: int, col: int) -> List[int]:
        if board[row][col] != EMPTY_CELL:
            return [board[row][col]]
        domain = []
        for num in range(1, 10):
            if SudokuSolver.is_valid_placement(board, row, col, num):
                domain.append(num)
        return domain
    @staticmethod
    def solve(board: List[List[int]], algorithm: str, callback: Optional[Callable] = None) -> bool:
        if algorithm == 'backtracking':
            solver = BacktrackingSolver()
        else:
            solver = ArcConsistencySolver()
        return solver.solve(board, callback)
    
    @staticmethod
    def generate_puzzle(difficulty='medium') -> List[List[int]]:
        board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        for box in range(3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            for i in range(3):
                for j in range(3):
                    board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]
        
        solver = BacktrackingSolver()
        if not solver.solve(board):
            return SudokuSolver.generate_puzzle(difficulty)
        cells_to_remove = {'easy': 35, 'medium': 45, 'hard': 52}.get(difficulty, 45)
    
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(cells)
        
        removed = 0
        for r, c in cells:
            if removed >= cells_to_remove:
                break
        
            backup = board[r][c]
            board[r][c] = EMPTY_CELL
            removed += 1
        
        return board


class BacktrackingSolver:
    
    def __init__(self):
        self.callback_count = 0
        self.max_callbacks = 50000
    
    def solve(self, board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        self.callback_count = 0
        return self._solve_recursive(board, callback)
    
    def validate_solvable(self, board: List[List[int]]) -> bool:
        board_copy = [row[:] for row in board]
        return self._solve_recursive(board_copy, callback=None)
    
    def _solve_recursive(self, board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if board[row][col] == EMPTY_CELL:
                    domain = SudokuSolver.get_domain(board, row, col)
                    
                    if len(domain) == 0:
                        return False
                    
                    for num in domain:
                        board[row][col] = num
                        
                        if callback and self.callback_count < self.max_callbacks:
                            callback(row, col, num, 'backtracking', domain.copy())
                            self.callback_count += 1
                        
                        if self._solve_recursive(board, callback):
                            return True
                        
                        board[row][col] = EMPTY_CELL
                    
                    return False
        
        return True


class ArcConsistencySolver:
    
    def __init__(self):
        self.domains: Dict[Tuple[int, int], List[int]] = {}
        self.arcs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.callback_count = 0
        self.max_callbacks = 50000
        self.constraints_tree: Dict[Tuple[int, int], Dict] = {}
    
    def solve(self, board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        self._initialize_domains(board)
        
        self._define_arcs()
        
        if not self._ac3(board, callback):
            return False
        
        self._update_grid_from_domains(board, callback)
        
        return self._solve_with_ac3(board, callback)
    
    def _initialize_domains(self, board: List[List[int]]):
        self.domains = {}
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if board[row][col] != EMPTY_CELL:
                    self.domains[(row, col)] = [board[row][col]]
                else:
                    self.domains[(row, col)] = list(range(1, 10))
                    self.domains[(row, col)] = SudokuSolver.get_domain(board, row, col)
    
    def _define_arcs(self):
        self.arcs = []
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell = (row, col)
                neighbors = self._get_neighbors(row, col)
                for neighbor in neighbors:
                    self.arcs.append((cell, neighbor))
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = set()
        
        for c in range(GRID_SIZE):
            if c != col:
                neighbors.add((row, c))
        
        for r in range(GRID_SIZE):
            if r != row:
                neighbors.add((r, col))
        
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if r != row or c != col:
                    neighbors.add((r, c))
        
        return list(neighbors)

    def build_constraints_tree(self, board: List[List[int]]) -> Dict[Tuple[int, int], Dict]:
        self._initialize_domains(board)
        self._define_arcs()

        tree: Dict[Tuple[int, int], Dict] = {}
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                key = (row, col)
                domain = self.domains.get(key, []).copy()
                tree[key] = {'domain': domain}

        self.constraints_tree = tree
        return tree

    def get_constraints_tree(self) -> Dict[Tuple[int, int], Dict]:
        return self.constraints_tree
    
    def _revise(self, board: List[List[int]], xi: Tuple[int, int], xj: Tuple[int, int]) -> bool:
        revised = False
        values_to_remove = []
        
        for value_i in self.domains[xi]:
            consistent = False
            
            if len(self.domains[xj]) == 1:
                value_j = self.domains[xj][0]
                if value_i != value_j:
                    consistent = True
            else:
                consistent = True

            if not consistent:
                values_to_remove.append(value_i)
                revised = True
        
        for value in values_to_remove:
            self.domains[xi].remove(value)
        
        return revised
    
    def _ac3(self, board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        queue = list(self.arcs)
    
        while queue:
            xi, xj = queue.pop(0)
            
            if self._revise(board, xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                neighbors = self._get_neighbors(xi[0], xi[1])
                for xk in neighbors:
                    if xk != xj:
                        queue.append((xk, xi))
        
        return True
    
    def _update_grid_from_domains(self, board: List[List[int]], callback: Optional[Callable] = None):
        updated = True
        while updated:
            updated = False
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if board[row][col] == EMPTY_CELL and len(self.domains[(row, col)]) == 1:
                        value = self.domains[(row, col)][0]
                        board[row][col] = value
                        updated = True
                        
                        if callback and self.callback_count < self.max_callbacks:
                            callback(row, col, value, 'arc-consistency', [value])
                            self.callback_count += 1
    
    def _solve_with_ac3(self, board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        all_filled = all(board[r][c] != EMPTY_CELL for r in range(GRID_SIZE) for c in range(GRID_SIZE))
        if all_filled:
            return True
        
        min_domain_size = float('inf')
        best_cell = None
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if board[row][col] == EMPTY_CELL:
                    domain_size = len(self.domains[(row, col)])
                    if domain_size < min_domain_size:
                        min_domain_size = domain_size
                        best_cell = (row, col)
        
        if best_cell is None:
            return True
        
        row, col = best_cell
        domain = self.domains[(row, col)].copy()
        
        for num in domain:
            if SudokuSolver.is_valid_placement(board, row, col, num):
                board[row][col] = num
                
                old_domains = {k: v.copy() for k, v in self.domains.items()}
                self.domains[(row, col)] = [num]
                
                if callback and self.callback_count < self.max_callbacks:
                    callback(row, col, num, 'arc-consistency', domain.copy())
                    self.callback_count += 1
                
                if self._ac3(board, callback):
                    self._update_grid_from_domains(board, callback)
                    
                    if self._solve_with_ac3(board, callback):
                        return True
                board[row][col] = EMPTY_CELL
                self.domains = old_domains
        
        return False


class SudokuCell(tk.Frame):
    
    def __init__(self, parent, row, col, value, is_editable, on_change, on_click):
        super().__init__(parent, bg=COLOR_GRID_LINE)
        self.row = row
        self.col = col
        self.is_editable = is_editable
        self.on_change = on_change
        self.on_click = on_click
        self._suppress_var = False
        self._last_value = value
        
        border_right = 3 if (col + 1) % 3 == 0 and col != 8 else 1
        border_bottom = 3 if (row + 1) % 3 == 0 and row != 8 else 1
        
        self.config(highlightthickness=0, bd=0)
        self.grid(padx=(0, border_right - 1), pady=(0, border_bottom - 1))
        
        self.var = tk.StringVar()
        self.entry = tk.Entry(
            self,
            textvariable=self.var,
            width=2,
            font=('Arial', 20, 'bold'),
            justify='center',
            bd=1,
            relief='solid',
            highlightthickness=2,
            highlightcolor=COLOR_PRIMARY,
            highlightbackground=COLOR_GRID_LINE
        )
        self.entry.pack(fill='both', expand=True)

        self._trace_name = None
        if is_editable:
            self._trace_name = self.var.trace_add('write', self._on_var_change)
            self.entry.bind('<Button-1>', lambda e: on_click(row, col))
            self.entry.config(cursor='hand2')
        else:
            self.entry.config(state='readonly', cursor='arrow')

        self.set_value(value)
        self.set_state('initial' if value != 0 else 'user')
    
    def _on_key(self, event):
        if not self.is_editable:
            return 'break'
        
        if event.char in '123456789':
            self.on_change(self.row, self.col, int(event.char))
            return 'break'
        elif event.keysym in ('BackSpace', 'Delete') or event.char == '0':
            self.on_change(self.row, self.col, 0)
            return 'break'
        return 'break'

    def _on_var_change(self, *args):
        """Handle changes to the Entry's StringVar, validating and notifying parent."""
        if self._suppress_var:
            return

        s = self.var.get()

        if s == '':
            value = 0
        elif len(s) == 1 and s in '123456789':
            value = int(s)
        else:
            self._suppress_var = True
            if self._last_value == 0:
                self.var.set('')
            else:
                self.var.set(str(self._last_value))
            self._suppress_var = False
            return

        self._last_value = value
        self.on_change(self.row, self.col, value)
    
    def set_value(self, value):
        self._suppress_var = True
        if value == 0:
            self.var.set('')
        else:
            self.var.set(str(value))
        self._last_value = value
        self._suppress_var = False
        if not self.is_editable:
            self.entry.config(state='readonly')
        else:
            self.entry.config(state='normal')

    def set_editable(self, is_editable):
        """Enable or disable editing for this cell, updating trace and state."""
        if is_editable == self.is_editable:
            return
        self.is_editable = is_editable
        if is_editable:
            if not self._trace_name:
                self._trace_name = self.var.trace_add('write', self._on_var_change)
            self.entry.config(state='normal', cursor='hand2')
        else:
            if self._trace_name:
                try:
                    self.var.trace_remove('write', self._trace_name)
                except Exception:
                    pass
                self._trace_name = None
            self.entry.config(state='readonly', cursor='arrow')
    
    def set_state(self, state):
        colors = {
            'initial': (COLOR_INITIAL, '#1e293b'),
            'user': (COLOR_USER, COLOR_PRIMARY),
            'ai-solving': (COLOR_AI_SOLVING, '#92400e'),
            'ai-solved': (COLOR_BG, COLOR_AI_SOLVED),
            'error': (COLOR_ERROR, '#dc2626')
        }
        bg, fg = colors.get(state, (COLOR_BG, '#000000'))
        self.entry.config(bg=bg, fg=fg)
    
    def set_selected(self, selected):
        if selected:
            self.entry.config(highlightbackground=COLOR_PRIMARY, highlightthickness=2)
        else:
            self.entry.config(highlightbackground=COLOR_GRID_LINE, highlightthickness=2)


class SudokuPuzzleSolver:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Puzzle Solver")
        self.root.geometry("1400x800")
        self.root.configure(bg=COLOR_BG)
        
        self.mode = 'puzzle'
        self.algorithm = 'backtracking'
        self.difficulty = 'medium'
        self.initial_board = SudokuSolver.generate_puzzle(self.difficulty)
        self.current_board = [row[:] for row in self.initial_board]
        self.editable_board = [[cell == EMPTY_CELL for cell in row] for row in self.initial_board]
        self.selected_cell = None
        self.is_solving = False
        self.is_paused = False
        self.steps_count = 0
        self.elapsed_time = 0
        self.is_solved = False
        self.solving_steps = []
        self.current_step = 0
        self.start_time = 0
        
        self.comparison_data = {
            'easy': {'backtracking': [], 'arc-consistency': []},
            'medium': {'backtracking': [], 'arc-consistency': []},
            'hard': {'backtracking': [], 'arc-consistency': []}
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI"""
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        title_frame = tk.Frame(main_frame, bg=COLOR_BG)
        title_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(
            title_frame,
            text="Sudoku Puzzle Solver",
            font=('Arial', 32, 'bold'),
            bg=COLOR_BG,
            fg='#1e293b'
        ).pack()
        
        tk.Label(
            title_frame,
            text="Watch AI solve puzzles or input your own",
            font=('Arial', 12),
            bg=COLOR_BG,
            fg='#64748b'
        ).pack()
        
        content_frame = tk.Frame(main_frame, bg=COLOR_BG)
        content_frame.pack(fill='both', expand=True)
        
        left_panel = tk.Frame(content_frame, bg=COLOR_BG)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 20))
        
        self.grid_frame = tk.Frame(left_panel, bg=COLOR_GRID_THICK, bd=3, relief='solid')
        self.grid_frame.pack(pady=(0, 20))
        
        self.cells = []
        for row in range(GRID_SIZE):
            cell_row = []
            for col in range(GRID_SIZE):
                cell = SudokuCell(
                    self.grid_frame,
                    row, col,
                    self.current_board[row][col],
                    self.editable_board[row][col],
                    self.on_cell_change,
                    self.on_cell_click
                )
                cell.grid(row=row, column=col, sticky='nsew')
                cell_row.append(cell)
            self.cells.append(cell_row)
        
        self.setup_control_panel(left_panel)
        
        middle_panel = tk.Frame(content_frame, bg=COLOR_BG, width=350)
        middle_panel.pack(side='left', fill='both', padx=(0, 20))
        middle_panel.pack_propagate(False)
        
        self.setup_progress_panel(middle_panel)
        
        right_panel = tk.Frame(content_frame, bg=COLOR_BG, width=300)
        right_panel.pack(side='left', fill='both', expand=True)
        
        self.setup_steps_viewer(right_panel)
    
    def setup_control_panel(self, parent):
        """Setup control panel UI"""
        panel = tk.LabelFrame(
            parent,
            text="Controls",
            font=('Arial', 12, 'bold'),
            bg=COLOR_BG,
            fg='#1e293b',
            bd=2,
            relief='solid',
            padx=15,
            pady=15
        )
        panel.pack(fill='x')

        mode_frame = tk.Frame(panel, bg=COLOR_BG)
        mode_frame.pack(fill='x', pady=(0, 10))
        tk.Label(mode_frame, text="Mode", font=('Arial', 10, 'bold'), bg=COLOR_BG).pack(anchor='w')
        modes_buttons = tk.Frame(mode_frame, bg=COLOR_BG)
        modes_buttons.pack(fill='x', pady=(5, 0))

        def _style_mode(btn, active=False):
            if active:
                btn.config(bg=COLOR_PRIMARY, fg='white')
            else:
                btn.config(bg='#e2e8f0', fg='#1e293b')
            btn.config(font=('Arial', 9, 'bold'), relief='flat', bd=0, padx=8, pady=6, cursor='hand2')

        self.mode_puzzle_btn = tk.Button(modes_buttons, text='🎲 Generated', command=lambda: self.set_mode('puzzle'))
        _style_mode(self.mode_puzzle_btn, active=(self.mode == 'puzzle'))
        self.mode_puzzle_btn.pack(side='left')

        self.mode_user_btn = tk.Button(modes_buttons, text='✍️ User Input', command=lambda: self.set_mode('user-input'))
        _style_mode(self.mode_user_btn, active=(self.mode == 'user-input'))
        self.mode_user_btn.pack(side='left', padx=(8, 0))

        
        difficulty_frame = tk.Frame(panel, bg=COLOR_BG)
        difficulty_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(difficulty_frame, text="Difficulty", font=('Arial', 10, 'bold'), bg=COLOR_BG).pack(anchor='w')
        difficulty_buttons = tk.Frame(difficulty_frame, bg=COLOR_BG)
        difficulty_buttons.pack(fill='x', pady=(5, 0))
        
        self.easy_btn = tk.Button(
            difficulty_buttons,
            text="Easy",
            command=lambda: self.set_difficulty('easy'),
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.easy_btn.pack(side='left', fill='x', expand=True, padx=(0, 3))
        
        self.medium_btn = tk.Button(
            difficulty_buttons,
            text="Medium",
            command=lambda: self.set_difficulty('medium'),
            bg=COLOR_PRIMARY,
            fg='white',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.medium_btn.pack(side='left', fill='x', expand=True, padx=(0, 3))
        
        self.hard_btn = tk.Button(
            difficulty_buttons,
            text="Hard",
            command=lambda: self.set_difficulty('hard'),
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.hard_btn.pack(side='left', fill='x', expand=True)
        
        algo_frame = tk.Frame(panel, bg=COLOR_BG)
        algo_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(algo_frame, text="Algorithm", font=('Arial', 10, 'bold'), bg=COLOR_BG).pack(anchor='w')
        algo_buttons = tk.Frame(algo_frame, bg=COLOR_BG)
        algo_buttons.pack(fill='x', pady=(5, 0))
        
        self.backtrack_btn = tk.Button(
            algo_buttons,
            text="🔄 Backtracking",
            command=lambda: self.set_algorithm('backtracking'),
            bg=COLOR_PRIMARY,
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.backtrack_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.arc_btn = tk.Button(
            algo_buttons,
            text="🧠 Arc Consistency",
            command=lambda: self.set_algorithm('arc-consistency'),
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.arc_btn.pack(side='left', fill='x', expand=True)
        
        action_frame = tk.Frame(panel, bg=COLOR_BG)
        action_frame.pack(fill='x')
        
        tk.Button(
            action_frame,
            text="✓ Validate",
            command=self.validate_puzzle,
            bg='#3b82f6',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        ).pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.solve_btn = tk.Button(
            action_frame,
            text="▶ Start Solving",
            command=self.start_solving,
            bg='#10b981',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        )
        self.solve_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        tk.Button(
            action_frame,
            text="↻ Reset",
            command=self.reset_puzzle,
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        ).pack(side='left', padx=(0, 5))
        
        tk.Button(
            action_frame,
            text="New Puzzle",
            command=self.new_puzzle,
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 11, 'bold'),
            relief='flat',
            cursor='hand2',
            height=2
        ).pack(side='left')
    
    def setup_progress_panel(self, parent):
        """Setup progress panel UI"""
        panel = tk.LabelFrame(
            parent,
            text="⚡ Solving Progress",
            font=('Arial', 12, 'bold'),
            bg=COLOR_BG,
            fg='#1e293b',
            bd=2,
            relief='solid',
            padx=15,
            pady=15
        )
        panel.pack(fill='both', expand=True)
        
        self.timer_label = tk.Label(
            panel,
            text="⏱ 0:00",
            font=('Arial', 10),
            bg=COLOR_BG,
            fg='#64748b'
        )
        self.timer_label.pack(anchor='e')
        
        progress_frame = tk.Frame(panel, bg=COLOR_BG)
        progress_frame.pack(fill='x', pady=(10, 0))
        
        tk.Label(progress_frame, text="Cells Filled", font=('Arial', 10), bg=COLOR_BG).pack(anchor='w')
        
        self.progress_label = tk.Label(
            progress_frame,
            text="36 / 81",
            font=('Arial', 10, 'bold'),
            bg=COLOR_BG
        )
        self.progress_label.pack(anchor='e')
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(fill='x', pady=(5, 0))
        
        stats_frame = tk.Frame(panel, bg=COLOR_BG)
        stats_frame.pack(fill='x', pady=(15, 0))
        
        stat1 = tk.Frame(stats_frame, bg='#f1f5f9', relief='solid', bd=1)
        stat1.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.steps_label = tk.Label(
            stat1,
            text="0",
            font=('Arial', 24, 'bold'),
            bg='#f1f5f9',
            fg=COLOR_PRIMARY
        )
        self.steps_label.pack(pady=(10, 0))
        
        tk.Label(
            stat1,
            text="Steps Taken",
            font=('Arial', 9),
            bg='#f1f5f9',
            fg='#64748b'
        ).pack(pady=(0, 10))
        
        stat2 = tk.Frame(stats_frame, bg='#f1f5f9', relief='solid', bd=1)
        stat2.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        self.complete_label = tk.Label(
            stat2,
            text="44.4%",
            font=('Arial', 24, 'bold'),
            bg='#f1f5f9',
            fg=COLOR_PRIMARY
        )
        self.complete_label.pack(pady=(10, 0))
        
        tk.Label(
            stat2,
            text="Complete",
            font=('Arial', 9),
            bg='#f1f5f9',
            fg='#64748b'
        ).pack(pady=(0, 10))
        
        self.update_progress()
    
    def setup_steps_viewer(self, parent):
        """Setup steps viewer UI"""
        panel = tk.LabelFrame(
            parent,
            text="📋 Solving Steps",
            font=('Arial', 12, 'bold'),
            bg=COLOR_BG,
            fg='#1e293b',
            bd=2,
            relief='solid'
        )
        panel.pack(fill='both', expand=True)
        
        self.steps_badge = tk.Label(
            panel,
            text="0 steps",
            font=('Arial', 9),
            bg='#e2e8f0',
            fg='#1e293b',
            padx=8,
            pady=2
        )
        self.steps_badge.place(relx=1.0, x=-10, y=5, anchor='ne')
        
        canvas = tk.Canvas(panel, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient='vertical', command=canvas.yview)
        self.steps_frame = tk.Frame(canvas, bg=COLOR_BG)
        
        self.steps_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        canvas.create_window((0, 0), window=self.steps_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
    
        self.no_steps_label = tk.Label(
            self.steps_frame,
            text="📋\n\nNo steps yet\nStart solving to see the steps",
            font=('Arial', 11),
            bg=COLOR_BG,
            fg='#94a3b8',
            justify='center'
        )
        self.no_steps_label.pack(pady=50)

        constraints_panel = tk.LabelFrame(
            parent,
            text="🌳 Constraints Tree",
            font=('Arial', 12, 'bold'),
            bg=COLOR_BG,
            fg='#1e293b',
            bd=2,
            relief='solid'
        )
        constraints_panel.pack(fill='both', expand=True, pady=(10, 0))

        btn_row = tk.Frame(constraints_panel, bg=COLOR_BG)
        btn_row.pack(fill='x', padx=8, pady=(8, 0))

        tk.Button(
            btn_row,
            text="Refresh",
            command=self.refresh_constraints_tree,
            bg=COLOR_PRIMARY,
            fg='white',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2'
        ).pack(side='left')

        tk.Button(
            btn_row,
            text="Export",
            command=self.export_constraints_tree,
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2'
        ).pack(side='left', padx=(8,0))

        tree_frame = tk.Frame(constraints_panel, bg=COLOR_BG)
        tree_frame.pack(fill='both', expand=True, padx=8, pady=8)

        self.constraints_text = tk.Text(tree_frame, wrap='none', height=12, bg='#f8fafc')
        self.constraints_text.pack(side='left', fill='both', expand=True)

        tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.constraints_text.yview)
        tree_scroll.pack(side='right', fill='y')
        self.constraints_text.configure(yscrollcommand=tree_scroll.set)

        self.refresh_constraints_tree()
    
    def set_difficulty(self, difficulty):
        """Set puzzle difficulty"""
        self.difficulty = difficulty
        
        buttons = {
            'easy': self.easy_btn,
            'medium': self.medium_btn,
            'hard': self.hard_btn
        }
        
        for diff, btn in buttons.items():
            if diff == difficulty:
                btn.config(bg=COLOR_PRIMARY, fg='white')
            else:
                btn.config(bg='#e2e8f0', fg='#1e293b')
    
    def set_algorithm(self, algorithm):
        """Set solving algorithm"""
        self.algorithm = algorithm
        if algorithm == 'backtracking':
            self.backtrack_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.arc_btn.config(bg='#e2e8f0', fg='#1e293b')
        else:
            self.arc_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.backtrack_btn.config(bg='#e2e8f0', fg='#1e293b')

    def set_mode(self, mode: str):
        """Switch between 'puzzle' (generated) and 'user-input' (empty board).

        Updates `initial_board`, `current_board`, and `editable_board`, then refreshes UI.
        """
        if mode == self.mode:
            return
        self.mode = mode
        if mode == 'user-input':
            self.initial_board = [[EMPTY_CELL for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            self.current_board = [row[:] for row in self.initial_board]
            self.editable_board = [[True for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        else:
            self.initial_board = SudokuSolver.generate_puzzle(self.difficulty)
            self.current_board = [row[:] for row in self.initial_board]
            self.editable_board = [[cell == EMPTY_CELL for cell in row] for row in self.initial_board]

        self.reset_state()
        self.refresh_grid()
        try:
            if hasattr(self, 'mode_puzzle_btn') and hasattr(self, 'mode_user_btn'):
                if self.mode == 'puzzle':
                    self.mode_puzzle_btn.config(bg=COLOR_PRIMARY, fg='white')
                    self.mode_user_btn.config(bg='#e2e8f0', fg='#1e293b')
                else:
                    self.mode_user_btn.config(bg=COLOR_PRIMARY, fg='white')
                    self.mode_puzzle_btn.config(bg='#e2e8f0', fg='#1e293b')
        except Exception:
            pass
        
    
    def new_puzzle(self):
        """Generate a new puzzle"""
        if self.mode == 'user-input':
            self.initial_board = [[EMPTY_CELL for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            self.editable_board = [[True for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        else:
            self.initial_board = SudokuSolver.generate_puzzle(self.difficulty)
            self.editable_board = [[cell == EMPTY_CELL for cell in row] for row in self.initial_board]

        self.current_board = [row[:] for row in self.initial_board]
        self.reset_state()
        self.refresh_grid()
    
    def reset_puzzle(self):
        """Reset to initial state"""
        self.current_board = [row[:] for row in self.initial_board]
        self.reset_state()
        self.refresh_grid()
    
    def reset_state(self):
        """Reset solving state"""
        self.is_solving = False
        self.is_paused = False
        self.steps_count = 0
        self.elapsed_time = 0
        self.is_solved = False
        self.solving_steps = []
        self.current_step = 0
        self.update_progress()
        self.update_steps_viewer()
    
    def refresh_grid(self):
        """Refresh the grid display"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self.cells[row][col].set_value(self.current_board[row][col])
                state = 'initial' if self.current_board[row][col] != 0 and not self.editable_board[row][col] else 'user'
                self.cells[row][col].set_state(state)
                self.cells[row][col].set_editable(self.editable_board[row][col])
    
    def on_cell_change(self, row, col, value):
        """Handle cell value change"""
        if not self.editable_board[row][col] or self.is_solving:
            return
        
        self.current_board[row][col] = value
        
        if value != 0 and not SudokuSolver.is_valid_placement(self.current_board, row, col, value):
            self.cells[row][col].set_state('error')
            self.root.after(500, lambda: self.cells[row][col].set_state('user'))
            messagebox.showwarning("Invalid Move", f"The value {value} cannot be placed at cell ({row + 1}, {col + 1}).\nIt conflicts with existing values.")
            self.current_board[row][col] = 0
            self.cells[row][col].set_value(0)
            return
        
        self.cells[row][col].set_value(value)
        self.cells[row][col].set_state('user')
        self.update_progress()
    
    def on_cell_click(self, row, col):
        """Handle cell click"""
        if self.selected_cell:
            old_row, old_col = self.selected_cell
            self.cells[old_row][old_col].set_selected(False)
        
        self.selected_cell = (row, col)
        self.cells[row][col].set_selected(True)
    
    def start_solving(self):
        """Start the solving process"""
        if self.is_solving:
            self.is_paused = not self.is_paused
            self.solve_btn.config(text="⏸ Pause" if not self.is_paused else "▶ Resume")
            return
        
        self.solve_btn.config(text="🔍 Validating...", state='disabled')
        self.root.update()
        
        board_copy = [row[:] for row in self.current_board]
        validator = BacktrackingSolver()
        
        if not validator.validate_solvable(board_copy):
            messagebox.showerror("Invalid Puzzle", "This puzzle is not solvable!\nPlease check your input or generate a new puzzle.")
            self.solve_btn.config(text="▶ Start Solving", state='normal', bg='#10b981')
            return
        
        self.solve_btn.config(text="⏳ Processing...", state='disabled')
        self.root.update()
        
        board_copy = [row[:] for row in self.current_board]
        self.solving_steps = []
        
        def step_callback(row, col, value, method, domain):
            self.solving_steps.append((row, col, value, method, domain))
        
        try:
            solved = SudokuSolver.solve(board_copy, self.algorithm, step_callback)
        except Exception as e:
            messagebox.showerror("Error", f"Solving failed: {str(e)}")
            self.solve_btn.config(text="▶ Start Solving", state='normal', bg='#10b981')
            return
        
        self.solve_btn.config(state='normal')
        
        if not solved:
            messagebox.showerror("Unsolvable", "This puzzle cannot be solved or took too long!\nTry generating a new puzzle.")
            self.solve_btn.config(text="▶ Start Solving", bg='#10b981')
            return
        
        if len(self.solving_steps) == 0:
            messagebox.showinfo("Already Solved", "This puzzle is already complete!")
            self.solve_btn.config(text="▶ Start Solving", bg='#10b981')
            return
        
        self.is_solving = True
        self.current_step = 0
        self.steps_count = 0
        self.start_time = time.time()
        self.solve_btn.config(text="⏸ Pause", bg='#f59e0b')
        
        self.update_steps_viewer()
        self.animate_solving()
    
    def animate_solving(self):
        """Animate the solving process"""
        if not self.is_solving or self.current_step >= len(self.solving_steps):
            if self.current_step >= len(self.solving_steps):
                self.is_solving = False
                self.is_solved = True
                self.solve_btn.config(text="▶ Start Solving", bg='#10b981')
                messagebox.showinfo("Solved!", f"Puzzle solved in {self.steps_count} steps and {self.elapsed_time:.2f} seconds!")
            return
        
        if self.is_paused:
            self.root.after(100, self.animate_solving)
            return
        
        row, col, value, method, domain = self.solving_steps[self.current_step]
        
        self.current_board[row][col] = value
        self.cells[row][col].set_value(value)
        self.cells[row][col].set_state('ai-solving')
        
        self.root.after(150, lambda: self.cells[row][col].set_state('ai-solved'))
        
        self.steps_count += 1
        self.current_step += 1
        
        self.update_progress()
        self.update_steps_viewer()
        self.update_timer()
        
        self.root.after(200, self.animate_solving)
    
    def update_progress(self):
        """Update progress indicators"""
        filled = sum(1 for row in self.current_board for cell in row if cell != 0)
        total = GRID_SIZE * GRID_SIZE
        percentage = (filled / total) * 100
        
        self.progress_label.config(text=f"{filled} / {total}")
        self.progress_bar['value'] = percentage
        self.steps_label.config(text=str(self.steps_count))
        self.complete_label.config(text=f"{percentage:.1f}%")
    
    def update_timer(self):
        """Update the timer"""
        if self.is_solving and not self.is_paused:
            self.elapsed_time = time.time() - self.start_time
            minutes = int(self.elapsed_time // 60)
            seconds = int(self.elapsed_time % 60)
            self.timer_label.config(text=f"⏱ {minutes}:{seconds:02d}")
            self.root.after(100, self.update_timer)
    
    def update_steps_viewer(self):
        for widget in self.steps_frame.winfo_children():
            widget.destroy()
        
        self.steps_badge.config(text=f"{len(self.solving_steps)} steps")
        
        if not self.solving_steps:
            self.no_steps_label = tk.Label(
                self.steps_frame,
                text="📋\n\nNo steps yet\nStart solving to see the steps",
                font=('Arial', 11),
                bg=COLOR_BG,
                fg='#94a3b8',
                justify='center'
            )
            self.no_steps_label.pack(pady=50)
            return
        
        steps_to_show = self.solving_steps[-50:] if len(self.solving_steps) > 50 else self.solving_steps
        start_index = len(self.solving_steps) - len(steps_to_show)
        
        for idx, (row, col, value, method, domain) in enumerate(steps_to_show):
            i = start_index + idx
            is_current = i == self.current_step - 1
            is_past = i < self.current_step
            
            bg_color = COLOR_PRIMARY if is_current else '#f1f5f9' if is_past else '#f8fafc'
            fg_color = 'white' if is_current else '#1e293b'
            
            step_frame = tk.Frame(self.steps_frame, bg=bg_color, relief='solid', bd=2 if is_current else 1)
            step_frame.pack(fill='x', pady=2, padx=5)
            
            step_num_label = tk.Label(
                step_frame,
                text=str(i + 1),
                font=('Arial', 10, 'bold'),
                bg=bg_color,
                fg=fg_color,
                width=4,
                anchor='center'
            )
            step_num_label.pack(side='left', padx=5, pady=5)
            
            info_frame = tk.Frame(step_frame, bg=bg_color)
            info_frame.pack(side='left', fill='x', expand=True, padx=5)
            
            tk.Label(
                info_frame,
                text=f"Cell ({row + 1}, {col + 1})",
                font=('Arial', 9, 'bold'),
                bg=bg_color,
                fg=fg_color
            ).pack(anchor='w')
            
            domain_str = ', '.join(map(str, domain))
            tk.Label(
                info_frame,
                text=f"Value: {value} | Domain: [{domain_str}]",
                font=('Arial', 8),
                bg=bg_color,
                fg=fg_color
            ).pack(anchor='w')
            
            algo_text = "BT" if method == 'backtracking' else "AC3"
            algo_color = '#8b5cf6' if method == 'arc-consistency' else '#3b82f6'
            tk.Label(
                info_frame,
                text=algo_text,
                font=('Arial', 7, 'bold'),
                bg=fg_color if is_current else algo_color,
                fg=bg_color if is_current else 'white',
                padx=4,
                pady=1
            ).pack(anchor='w', pady=(2, 0))
            
            tk.Label(
                step_frame,
                text=str(value),
                font=('Arial', 16, 'bold'),
                bg=bg_color,
                fg=fg_color,
                width=2
            ).pack(side='right', padx=10, pady=5)

    def refresh_constraints_tree(self):
        """Rebuild and render the constraints tree for the current board."""
        try:
            ac_solver = ArcConsistencySolver()
            tree = ac_solver.build_constraints_tree(self.current_board)
            self.render_constraints_tree(tree)
        except Exception as e:
            self.constraints_text.delete('1.0', 'end')
            self.constraints_text.insert('end', f"Error building constraints tree: {e}")

    def render_constraints_tree(self, tree: Dict[Tuple[int, int], Dict]):
        """Render a readable representation of the constraints tree into the text widget."""
        self.constraints_text.config(state='normal')
        self.constraints_text.delete('1.0', 'end')

        lines = []
        lines.append(f"Constraints Tree - {len(tree)} nodes")
        lines.append('')

        for (r, c), node in sorted(tree.items()):
            domain = node.get('domain', [])
            domain_str = ','.join(map(str, domain)) if domain else '[]'
            lines.append(f"Cell ({r+1},{c+1}): domain=[{domain_str}]")

        self.constraints_text.insert('end', '\n'.join(lines))
        self.constraints_text.config(state='disabled')

    def export_constraints_tree(self):
        """Copy the current constraints tree text to the clipboard for export."""
        try:
            text = self.constraints_text.get('1.0', 'end').strip()
            if not text:
                messagebox.showinfo("Export", "No constraints to export.")
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Export", "Constraints tree copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def validate_puzzle(self):
        """Validate current puzzle for conflicts and solvability."""
        conflicts = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                val = self.current_board[row][col]
                if val != 0 and not SudokuSolver.is_valid_placement(self.current_board, row, col, val):
                    conflicts.append((row, col))

        if conflicts:
            for (r, c) in conflicts:
                self.cells[r][c].set_state('error')

            def _clear_conflicts(conf_list=conflicts):
                for (r, c) in conf_list:
                    state = 'initial' if (self.initial_board[r][c] != 0 and not self.editable_board[r][c]) else 'user'
                    self.cells[r][c].set_state(state)

            self.root.after(800, _clear_conflicts)
            messagebox.showerror("Invalid Puzzle", f"Found {len(conflicts)} conflicting cell(s). Please fix them before solving.")
            return

        validator = BacktrackingSolver()
        board_copy = [row[:] for row in self.current_board]
        solvable = validator.validate_solvable(board_copy)

        if solvable:
            messagebox.showinfo("Valid Puzzle", "Puzzle is valid and appears solvable.")
        else:
            messagebox.showwarning("Unsolvable Puzzle", "Puzzle has no solution (or is too constrained).")


if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuPuzzleSolver(root)
    root.mainloop()