import tkinter as tk
from tkinter import ttk, messagebox
import time
import random
from typing import List, Tuple, Optional, Callable

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
        # Check row
        for c in range(GRID_SIZE):
            if c != col and board[row][c] == num:
                return False
        
        # Check column
        for r in range(GRID_SIZE):
            if r != row and board[r][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r != row or c != col) and board[r][c] == num:
                    return False
        
        return True
    
    @staticmethod
    def solve_backtracking(board: List[List[int]], callback: Optional[Callable] = None) -> bool:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if board[row][col] == EMPTY_CELL:
                    for num in range(1, 10):
                        if SudokuSolver.is_valid_placement(board, row, col, num):
                            board[row][col] = num
                            if callback:
                                callback(row, col, num, 'backtracking')
                            
                            if SudokuSolver.solve_backtracking(board, callback):
                                return True
                            
                            board[row][col] = EMPTY_CELL
                    
                    return False
        return True
    
    @staticmethod
    def generate_puzzle(difficulty='medium') -> List[List[int]]:
        # Start with solved board
        board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        # Fill diagonal 3x3 boxes
        for box in range(3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            for i in range(3):
                for j in range(3):
                    board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]
        
        # Solve the board
        SudokuSolver.solve_backtracking(board)
        
        # Remove cells based on difficulty
        cells_to_remove = {'easy': 30, 'medium': 40, 'hard': 50}.get(difficulty, 40)
        
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        random.shuffle(cells)
        
        for r, c in cells[:cells_to_remove]:
            board[r][c] = EMPTY_CELL
        
        return board


class SudokuCell(tk.Frame):
    def __init__(self, parent, row, col, value, is_editable, on_change, on_click):
        super().__init__(parent, bg=COLOR_GRID_LINE)
        self.row = row
        self.col = col
        self.is_editable = is_editable
        self.on_change = on_change
        self.on_click = on_click
        
        # Border thickness
        border_right = 3 if (col + 1) % 3 == 0 and col != 8 else 1
        border_bottom = 3 if (row + 1) % 3 == 0 and row != 8 else 1
        
        self.config(
            highlightthickness=0,
            bd=0
        )
        self.grid(padx=(0, border_right - 1), pady=(0, border_bottom - 1))
        
        self.entry = tk.Entry(
            self,
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
        
        self.set_value(value)
        self.set_state('initial' if value != 0 else 'user')
        
        if is_editable:
            self.entry.bind('<KeyPress>', self._on_key)
            self.entry.bind('<Button-1>', lambda e: on_click(row, col))
            self.entry.config(cursor='hand2')
        else:
            self.entry.config(state='readonly', cursor='arrow')
    
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
    
    def set_value(self, value):
        self.entry.config(state='normal')
        self.entry.delete(0, 'end')
        if value != 0:
            self.entry.insert(0, str(value))
        if not self.is_editable:
            self.entry.config(state='readonly')
    
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
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Puzzle Solver")
        self.root.geometry("1400x800")
        self.root.configure(bg=COLOR_BG)
        
        # State variables
        self.mode = 'auto-solve'
        self.algorithm = 'backtracking'
        self.initial_board = SudokuSolver.generate_puzzle('medium')
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
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
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
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=COLOR_BG)
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Grid and controls
        left_panel = tk.Frame(content_frame, bg=COLOR_BG)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 20))
        
        # Sudoku Grid
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
        
        # Control Panel
        self.setup_control_panel(left_panel)
        
        # Middle panel - Progress
        middle_panel = tk.Frame(content_frame, bg=COLOR_BG, width=350)
        middle_panel.pack(side='left', fill='both', padx=(0, 20))
        middle_panel.pack_propagate(False)
        
        self.setup_progress_panel(middle_panel)
        
        # Right panel - Steps viewer
        right_panel = tk.Frame(content_frame, bg=COLOR_BG, width=300)
        right_panel.pack(side='left', fill='both', expand=True)
        
        self.setup_steps_viewer(right_panel)
    
    def setup_control_panel(self, parent):
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
        
        # Game Mode
        mode_frame = tk.Frame(panel, bg=COLOR_BG)
        mode_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(mode_frame, text="Game Mode", font=('Arial', 10, 'bold'), bg=COLOR_BG).pack(anchor='w')
        mode_buttons = tk.Frame(mode_frame, bg=COLOR_BG)
        mode_buttons.pack(fill='x', pady=(5, 0))
        
        self.auto_solve_btn = tk.Button(
            mode_buttons,
            text="⚡ Auto-Solve",
            command=lambda: self.set_mode('auto-solve'),
            bg=COLOR_PRIMARY,
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.auto_solve_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.user_input_btn = tk.Button(
            mode_buttons,
            text="📤 User Input",
            command=lambda: self.set_mode('user-input'),
            bg='#e2e8f0',
            fg='#1e293b',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2'
        )
        self.user_input_btn.pack(side='left', fill='x', expand=True)
        
        # Algorithm
        algo_frame = tk.Frame(panel, bg=COLOR_BG)
        algo_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(algo_frame, text="Algorithm", font=('Arial', 10, 'bold'), bg=COLOR_BG).pack(anchor='w')
        algo_buttons = tk.Frame(algo_frame, bg=COLOR_BG)
        algo_buttons.pack(fill='x', pady=(5, 0))
        
        self.backtrack_btn = tk.Button(
            algo_buttons,
            text="🧠 Backtracking",
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
        
        # Action buttons
        action_frame = tk.Frame(panel, bg=COLOR_BG)
        action_frame.pack(fill='x')
        
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
        
        # Timer
        self.timer_label = tk.Label(
            panel,
            text="⏱ 0:00",
            font=('Arial', 10),
            bg=COLOR_BG,
            fg='#64748b'
        )
        self.timer_label.pack(anchor='e')
        
        # Progress bar
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
        
        # Stats
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
        
        # Steps count badge
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
        
        # Scrollable frame
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
        
        # Initial message
        self.no_steps_label = tk.Label(
            self.steps_frame,
            text="📋\n\nNo steps yet\nStart solving to see the steps",
            font=('Arial', 11),
            bg=COLOR_BG,
            fg='#94a3b8',
            justify='center'
        )
        self.no_steps_label.pack(pady=50)
    
    def set_mode(self, mode):
        self.mode = mode
        if mode == 'auto-solve':
            self.auto_solve_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.user_input_btn.config(bg='#e2e8f0', fg='#1e293b')
            self.new_puzzle()
        else:
            self.user_input_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.auto_solve_btn.config(bg='#e2e8f0', fg='#1e293b')
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        if algorithm == 'backtracking':
            self.backtrack_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.arc_btn.config(bg='#e2e8f0', fg='#1e293b')
        else:
            self.arc_btn.config(bg=COLOR_PRIMARY, fg='white')
            self.backtrack_btn.config(bg='#e2e8f0', fg='#1e293b')
    
    def new_puzzle(self):
        self.initial_board = SudokuSolver.generate_puzzle('medium')
        self.current_board = [row[:] for row in self.initial_board]
        self.editable_board = [[cell == EMPTY_CELL for cell in row] for row in self.initial_board]
        self.reset_state()
        self.refresh_grid()
    
    def reset_puzzle(self):
        self.current_board = [row[:] for row in self.initial_board]
        self.reset_state()
        self.refresh_grid()
    
    def reset_state(self):
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
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self.cells[row][col].set_value(self.current_board[row][col])
                state = 'initial' if self.current_board[row][col] != 0 and not self.editable_board[row][col] else 'user'
                self.cells[row][col].set_state(state)
                self.cells[row][col].is_editable = self.editable_board[row][col]
    
    def on_cell_change(self, row, col, value):
        if not self.editable_board[row][col] or self.is_solving:
            return
        
        self.current_board[row][col] = value
        
        if value != 0 and not SudokuSolver.is_valid_placement(self.current_board, row, col, value):
            self.cells[row][col].set_state('error')
            self.root.after(500, lambda: self.cells[row][col].set_state('user'))
            return
        
        self.cells[row][col].set_value(value)
        self.cells[row][col].set_state('user')
        self.update_progress()
    
    def on_cell_click(self, row, col):
        if self.selected_cell:
            old_row, old_col = self.selected_cell
            self.cells[old_row][old_col].set_selected(False)
        
        self.selected_cell = (row, col)
        self.cells[row][col].set_selected(True)
    
    def start_solving(self):
        if self.is_solving:
            self.is_paused = not self.is_paused
            self.solve_btn.config(text="⏸ Pause" if not self.is_paused else "▶ Resume")
            return
        
        # Collect solving steps
        board_copy = [row[:] for row in self.current_board]
        self.solving_steps = []
        
        def step_callback(row, col, value, method):
            self.solving_steps.append((row, col, value, method))
        
        solved = SudokuSolver.solve_backtracking(board_copy, step_callback)
        
        if not solved:
            messagebox.showerror("Unsolvable", "This puzzle cannot be solved!")
            return
        
        self.is_solving = True
        self.current_step = 0
        self.steps_count = 0
        self.start_time = time.time()
        self.solve_btn.config(text="⏸ Pause", bg='#f59e0b')
        
        self.update_steps_viewer()
        self.animate_solving()
    
    def animate_solving(self):
        if not self.is_solving or self.current_step >= len(self.solving_steps):
            if self.current_step >= len(self.solving_steps):
                self.is_solving = False
                self.is_solved = True
                self.solve_btn.config(text="▶ Start Solving", bg='#10b981')
                messagebox.showinfo("Solved!", f"Puzzle solved in {self.steps_count} steps!")
            return
        
        if self.is_paused:
            self.root.after(100, self.animate_solving)
            return
        
        row, col, value, method = self.solving_steps[self.current_step]
        
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
        filled = sum(1 for row in self.current_board for cell in row if cell != 0)
        total = GRID_SIZE * GRID_SIZE
        percentage = (filled / total) * 100
        
        self.progress_label.config(text=f"{filled} / {total}")
        self.progress_bar['value'] = percentage
        self.steps_label.config(text=str(self.steps_count))
        self.complete_label.config(text=f"{percentage:.1f}%")
    
    def update_timer(self):
        if self.is_solving and not self.is_paused:
            self.elapsed_time = time.time() - self.start_time
            minutes = int(self.elapsed_time // 60)
            seconds = int(self.elapsed_time % 60)
            self.timer_label.config(text=f"⏱ {minutes}:{seconds:02d}")
            self.root.after(100, self.update_timer)
    
    def update_steps_viewer(self):
        # Clear existing steps
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
        
        for i, (row, col, value, method) in enumerate(self.solving_steps):
            is_current = i == self.current_step - 1
            is_past = i < self.current_step
            
            bg_color = COLOR_PRIMARY if is_current else '#f1f5f9' if is_past else '#f8fafc'
            fg_color = 'white' if is_current else '#1e293b'
            
            step_frame = tk.Frame(self.steps_frame, bg=bg_color, relief='solid', bd=2 if is_current else 1)
            step_frame.pack(fill='x', pady=2, padx=5)
            
            # Step number
            tk.Label(
                step_frame,
                text=str(i + 1),
                font=('Arial', 10, 'bold'),
                bg=bg_color,
                fg=fg_color,
                width=3
            ).pack(side='left', padx=5, pady=5)
            
            # Cell info
            info_frame = tk.Frame(step_frame, bg=bg_color)
            info_frame.pack(side='left', fill='x', expand=True, padx=5)
            
            tk.Label(
                info_frame,
                text=f"Cell ({row + 1}, {col + 1})",
                font=('Arial', 9, 'bold'),
                bg=bg_color,
                fg=fg_color
            ).pack(anchor='w')
            
            tk.Label(
                info_frame,
                text=f"Placed value: {value}",
                font=('Arial', 8),
                bg=bg_color,
                fg=fg_color
            ).pack(anchor='w')
            
            # Value display
            tk.Label(
                step_frame,
                text=str(value),
                font=('Arial', 16, 'bold'),
                bg=bg_color,
                fg=fg_color,
                width=2
            ).pack(side='right', padx=10, pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuPuzzleSolver(root)
    root.mainloop()