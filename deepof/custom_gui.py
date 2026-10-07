"""Custom GUI elements based on OpenCV, used for user interactions such as arena selection or skeleton definition."""

import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import networkx as nx
import numpy as np

import deepof.skeleton
from deepof.config import IMG_H_MAX, SCHEMA_IMAGE, SCHEMA_MATCH_RADIUS, SCHEMA_POSITIONS

def display_message(message: List[str]): # pragma: no cover
    """
    Opens a window that displays a message for the user

    Args:
        message: List of strings containing the message
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White color
    line_type = 2

    # Calculate dimensions based on message content
    max_line_length = max(len(line) for line in message)
    line_height = 30  # Height per line of text
    image_height = line_height * len(message) + 20  # Add some padding
    image_width = max(600, max_line_length * 12)  # Minimum width or based on longest line

    # Create a blank image with calculated dimensions
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Initial position for the first line of text
    x, y = 10, line_height

    # Loop through each line and put it on the image
    for line in message:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, line_type)
        y += line_height  # Move down for the next line

    window_name = "Arena scaling"

    # Display the image in a window
    cv2.imshow(window_name, image)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # Silently ignore if not supported

    try:
        # Wait for a key press or until the window is closed
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # Exit on 'q' key press
                break

            # Check if window is still open
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as e:
        print(f"An error occurred: {e}")   # Handle window close exception gracefully

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(window_name)


def confirm_action(message: str, window_name: str = "Confirm"):
    """
    Displays a confirmation dialog using OpenCV with multi-line support.
    
    Args:
        message: The message to display (use '\\n' for line breaks).
        window_name: Name of the OpenCV window.
    
    Returns:
        bool: True if 'y' pressed, False if 'n' pressed.
    """
    lines = message.split('\n')
    
    # Calculate image height based on number of lines
    line_height = 40
    padding = 100
    img_height = len(lines) * line_height + padding
    img_width = 800
    
    # Create black image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Add message lines
    y_pos = 50
    for line in lines:
        cv2.putText(img, line, (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height
    
    # Add instruction text at bottom
    cv2.putText(img, "Press 'y' to confirm, 'n' to cancel", (30, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    
    cv2.imshow(window_name, img)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # Silently ignore if not supported
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return False
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        

@dataclass
class DropdownConfig:
    # Position from right edge (will be calculated in init)
    margin_right: int = 10
    margin_top: int = 10
    width: int = 60  # Smaller width
    height: int = 25  # Smaller height
    option_height: int = 25  # Matching height
    font_scale: float = 0.5  # Smaller font
    font_thickness: int = 1
    border_color: Tuple[int, int, int] = (100, 100, 100)
    fill_color: Tuple[int, int, int] = (200, 200, 200)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    main_box_color: Tuple[int, int, int] = (220, 220, 220)  # Light gray background

class DropdownUI:
    def __init__(self, window_name: str, options: List[str], window_width: int, hidden: bool = False, config: DropdownConfig = None):
        self.window_name = window_name
        self.options = options
        self.config = config or DropdownConfig()
        
        # Calculate x position from right edge
        self.x = window_width - self.config.width - self.config.margin_right
        self.y = self.config.margin_top
        
        self.selected_option = options[0]
        self.is_open = False
        self.slider_value = 70
        self.slider_active = False
        self.hidden = hidden

    def _is_point_in_rect(self, point: Tuple[int, int], 
                         rect: Tuple[int, int, int, int]) -> bool:
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def draw(self, img: np.ndarray) -> None:
        if not self.hidden:
            cfg = self.config
            
            # Draw main box with background
            cv2.rectangle(img, 
                        (self.x, self.y),
                        (self.x + cfg.width, self.y + cfg.height),
                        cfg.main_box_color, -1)  # Filled rectangle
            cv2.rectangle(img, 
                        (self.x, self.y),
                        (self.x + cfg.width, self.y + cfg.height),
                        cfg.border_color, 1)  # Border
            
            # Calculate text size to center it
            text_size = cv2.getTextSize(self.selected_option, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    cfg.font_scale, 
                                    cfg.font_thickness)[0]
            text_x = self.x + (cfg.width - text_size[0]) // 2
            text_y = self.y + (cfg.height + text_size[1]) // 2
            
            cv2.putText(img, self.selected_option,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, cfg.text_color, cfg.font_thickness)
            
            if self.is_open:
                for i, option in enumerate(self.options):
                    y = self.y + cfg.height * (i + 1)
                    # Background
                    cv2.rectangle(img,
                                (self.x, y),
                                (self.x + cfg.width, y + cfg.option_height),
                                cfg.fill_color, -1)
                    # Border
                    cv2.rectangle(img,
                                (self.x, y),
                                (self.x + cfg.width, y + cfg.option_height),
                                cfg.border_color, 1)
                    # Centered text
                    text_size = cv2.getTextSize(option, 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            cfg.font_scale, 
                                            cfg.font_thickness)[0]
                    text_x = self.x + (cfg.width - text_size[0]) // 2
                    text_y = y + (cfg.option_height + text_size[1]) // 2
                    
                    cv2.putText(img, option,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            cfg.font_scale, cfg.text_color, cfg.font_thickness)

    def handle_mouse(self, event: int, x: int, y: int):
        """Returns the newly selected option if changed, None otherwise"""        
        if event != cv2.EVENT_LBUTTONDOWN or self.hidden:
            return None
            
        # Check main dropdown box click
        if self._is_point_in_rect((x, y), 
                                 (self.x, self.y, self.config.width, self.config.height)):
            self.is_open = not self.is_open
            return "Disable"
            
        if not self.is_open:
            return None
            
        # Check option clicks
        for i, option in enumerate(self.options):
            opt_y = self.y + self.config.height * (i + 1)
            if self._is_point_in_rect((x, y),
                                    (self.x, opt_y, self.config.width, self.config.option_height)):
                old_option = self.selected_option
                self.selected_option = option
                self.is_open = False
                return option if option != old_option else None
                
        self.is_open = False
        return None


class SkeletonEditor:
    """Window content and logic to define a custom skeleton on deepOF's mouse schema.

    The user places each body part of their labelling scheme on the schema. Body parts placed inside a yellow circle
    are treated as the corresponding deepOF body part, all others as additional body parts. Afterwards, a missing
    "Center" can be derived from other body parts, and the proposed graph can be edited.

    The class only handles drawing and input; the OpenCV window loop is in define_skeleton_gui.
    """

    PANEL_WIDTH = 340
    LINE_HEIGHT = 22
    YELLOW = (0, 215, 255)
    GREEN = (60, 170, 60)
    BLUE = (200, 110, 20)
    RED = (40, 40, 220)
    GREY = (150, 150, 150)
    EDGE = (180, 60, 160)

    def __init__(self, bodypart_names: List[str], skeleton: dict = None):
        """Initialize the editor.

        Args:
            bodypart_names (list): body part names of the tracking tables, in the order in which they are placed.
            skeleton (dict): optional skeleton (see deepof.skeleton) whose positions and derived points are loaded.
        """
        schema = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", SCHEMA_IMAGE), cv2.IMREAD_UNCHANGED)
        if schema.shape[2] == 4:  # transparent background to white
            alpha = schema[..., 3:] / 255.0
            schema = (schema[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        self.scale = min(1.0, IMG_H_MAX / schema.shape[0])
        self.schema = cv2.resize(schema, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        skeleton = skeleton or {}
        self.names = list(bodypart_names)
        self.positions = {n: tuple(p) for n, p in (skeleton.get("positions") or {}).items() if n in self.names}
        self.skipped = set()
        self.derive = dict(skeleton.get("derive") or {})
        self.history = []  # undo stack of ("place", name), ("skip", name), ("edge", a, b), ("center",)
        self.mode = "place"  # "place", "center" (question whether to derive a center) or "review"
        self.graph = None
        self.selected = None
        self.show_help = True
        if all(n in self.positions for n in self.names):
            self._finish_placement()

    # ---------------------------------------------------------------- state
    @property
    def current(self):
        """Name of the body part to place next, or None if all are placed or skipped."""
        return next((n for n in self.names if n not in self.positions and n not in self.skipped), None)

    def _resolve(self, graph: bool = True) -> dict:
        skeleton = {"positions": self.positions, "derive": self.derive}
        if graph and self.graph is not None:
            skeleton["graph"] = nx.to_dict_of_lists(self.graph)
        placed = [n for n in self.names if n in self.positions]
        if not graph:
            skeleton["graph"] = {n: [] for n in placed}  # no graph computation needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return deepof.skeleton.resolve_skeleton(skeleton, placed)

    def _matches(self) -> dict:
        return deepof.skeleton.match_to_deepof(self.positions)

    def _center_suggestion(self) -> List[str]:
        resolved = self._resolve(graph=False)
        return deepof.skeleton.suggest_derived_center(
            {n: p for n, p in resolved["positions"].items() if n in resolved["rename"].values()}
        )

    def _finish_placement(self):
        resolved = self._resolve(graph=False)
        if "Center" not in resolved["rename"].values() and "Center" not in self.derive and self._center_suggestion():
            self.mode = "center"
        else:
            self._start_review()

    def _start_review(self):
        self.graph = None
        self.graph = nx.Graph(self._resolve()["graph"])
        self.mode = "review"
        self.selected = None

    def to_skeleton(self) -> dict:
        """Return the skeleton (positions in schema pixels, derived points and graph in final names)."""
        return {
            "positions": {n: [float(v) for v in p] for n, p in self.positions.items()},
            "derive": self.derive,
            "graph": nx.to_dict_of_lists(self.graph) if self.graph is not None else None,
        }

    # ---------------------------------------------------------------- input
    def handle_click(self, x: int, y: int):
        """Handle a left click at window coordinates (x, y)."""
        if x >= self.schema.shape[1]:
            return
        sx, sy = x / self.scale, y / self.scale
        if self.mode == "place" and self.current is not None:
            name = self.current
            self.positions[name] = (round(sx, 1), round(sy, 1))
            self.history.append(("place", name))
            if self.current is None:
                self._finish_placement()
        elif self.mode == "review":
            positions = self._resolve()["positions"]
            node = min(self.graph.nodes, key=lambda n: np.hypot(positions[n][0] - sx, positions[n][1] - sy))
            if np.hypot(positions[node][0] - sx, positions[node][1] - sy) > 2 * SCHEMA_MATCH_RADIUS:
                self.selected = None
            elif self.selected is None or self.selected == node:
                self.selected = None if self.selected == node else node
            else:
                a, b = self.selected, node
                if self.graph.has_edge(a, b):
                    self.graph.remove_edge(a, b)
                else:
                    self.graph.add_edge(a, b)
                self.history.append(("edge", a, b))
                self.selected = None

    def handle_key(self, key: int) -> str:
        """Handle a key press. Returns "done" or "cancel" when the window should close, None otherwise."""
        if key == 27:  # Esc
            return "cancel"
        if key == ord("h"):
            self.show_help = not self.show_help
        elif key in (ord("d"), 8):  # d or Backspace: undo
            self._undo()
        elif self.mode == "place" and key == ord("s") and self.current is not None:
            self.skipped.add(self.current)
            self.history.append(("skip", self.current))
            if self.current is None:
                self._finish_placement()
        elif self.mode == "center" and key in (ord("y"), ord("n")):
            if key == ord("y"):
                self.derive["Center"] = self._center_suggestion()
            self.history.append(("center",))
            self._start_review()
        elif self.mode == "review" and key in (ord("q"), 13):
            if not nx.is_connected(self.graph):
                return None
            return "done"
        return None

    def _undo(self):
        if not self.history:
            return
        action = self.history.pop()
        if action[0] == "edge":
            a, b = action[1], action[2]
            if self.graph.has_edge(a, b):
                self.graph.remove_edge(a, b)
            else:
                self.graph.add_edge(a, b)
            return
        # undoing anything else leaves the review, as the graph depends on placement and derived points
        self.graph, self.selected = None, None
        self.history = [h for h in self.history if h[0] != "edge"]
        if action[0] == "center":
            self.derive.pop("Center", None)
            self.mode = "center"
            return
        self.mode = "place"
        if action[0] == "place":
            self.positions.pop(action[1], None)
        elif action[0] == "skip":
            self.skipped.discard(action[1])

    # ---------------------------------------------------------------- drawing
    def _text(self, img, text, pos, color=(0, 0, 0), scale=0.5, thickness=1, outline=False):
        if outline:  # readable on the dark mouse body
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _wrap(self, items: List[str], scale: float) -> List[str]:
        """Join items with commas into indented lines that fit into the panel."""
        lines, line = [], ""
        for i, item in enumerate(items):
            candidate = f"{line} {item}" if line else f"  {item}"
            candidate += "," if i < len(items) - 1 else ""
            if line and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > self.PANEL_WIDTH - 24:
                lines.append(line)
                candidate = f"  {item}" + ("," if i < len(items) - 1 else "")
            line = candidate
        return lines + ([line] if line else [])

    def render(self) -> np.ndarray:
        """Return the current window content as BGR image."""
        img = self.schema.copy()
        s = self.scale
        to_px = lambda p: (int(round(p[0] * s)), int(round(p[1] * s)))
        radius = max(4, int(round(SCHEMA_MATCH_RADIUS * s)))
        matches = self._matches()
        matched_targets = set(matches.values())

        if self.mode == "review":
            resolved = self._resolve()
            positions = resolved["positions"]
            for a, b in self.graph.edges:
                cv2.line(img, to_px(positions[a]), to_px(positions[b]), self.EDGE, 2, cv2.LINE_AA)
            for name in self.graph.nodes:
                color = self.RED if name in self.derive else (self.GREEN if name in SCHEMA_POSITIONS else self.BLUE)
                cv2.circle(img, to_px(positions[name]), radius // 2 + 2, color, -1, cv2.LINE_AA)
                if name == self.selected:
                    cv2.circle(img, to_px(positions[name]), radius + 4, self.RED, 2, cv2.LINE_AA)
                if name in self.derive or name not in SCHEMA_POSITIONS:
                    self._text(img, name, (to_px(positions[name])[0] + radius, to_px(positions[name])[1] - 4), color, 0.55, 1, True)
        else:
            for target, pos in SCHEMA_POSITIONS.items():
                cv2.circle(img, to_px(pos), radius, self.GREEN if target in matched_targets else self.YELLOW, 2, cv2.LINE_AA)
            for name, pos in self.positions.items():
                color = self.GREEN if name in matches else self.BLUE
                cv2.circle(img, to_px(pos), 4, color, -1, cv2.LINE_AA)
                if name not in matches:
                    self._text(img, name, (to_px(pos)[0] + 7, to_px(pos)[1] - 4), color, 0.55, 1, True)

        panel = np.full((img.shape[0], self.PANEL_WIDTH, 3), 245, dtype=np.uint8)
        y = 26
        if self.mode == "place":
            title = [f"Click where \"{self.current}\" is." if self.current else "All body parts placed."]
            info = ["Inside a yellow circle: treated as that",
                    "deepOF body part. Elsewhere: additional",
                    "body part, kept with its own name."]
        elif self.mode == "center":
            title = ["No \"Center\" body part found."]
            info = ["Derive it as the mean of:"] + self._wrap(self._center_suggestion(), 0.45) + ["y: yes   n: no"]
        else:
            title = ["Check the graph."]
            info = ["Click two body parts to add or remove", "their connection.",
                    "The graph must be connected." if not nx.is_connected(self.graph) else "q / Enter: save the skeleton."]
        for line in title:
            self._text(panel, line, (12, y), (0, 0, 0), 0.55, 2)
            y += self.LINE_HEIGHT + 4
        for line in info:
            self._text(panel, line, (12, y), (60, 60, 60), 0.45)
            y += self.LINE_HEIGHT - 2
        y += 10

        rename = self._resolve(graph=False)["rename"] if self.positions else {}
        for name in self.names:
            if name == self.current and self.mode == "place":
                cv2.rectangle(panel, (6, y - 16), (self.PANEL_WIDTH - 6, y + 6), self.YELLOW, -1)
                status, color = "<- click", (0, 0, 0)
            elif name in self.skipped:
                status, color = "skipped", self.GREY
            elif name in self.positions:
                final = rename.get(name, name)
                status = f"-> {final}" if name in matches else ("extra" if final == name else f"extra ({final})")
                color = self.GREEN if name in matches else self.BLUE
            else:
                status, color = "", (0, 0, 0)
            self._text(panel, name[:22], (14, y), color, 0.47)
            self._text(panel, status, (180, y), color, 0.47)
            y += self.LINE_HEIGHT
        for name, parts in self.derive.items():
            self._text(panel, name[:22], (14, y), self.RED, 0.47)
            self._text(panel, "derived (mean)", (180, y), self.RED, 0.47)
            y += self.LINE_HEIGHT

        if self.show_help:
            help_lines = (["s: skip (body part not on the mouse)"] if self.mode == "place" else []) + [
                "d / Backspace: undo", "h: hide help", "Esc: cancel"]
            y = img.shape[0] - 12 - self.LINE_HEIGHT * (len(help_lines) - 1)
            for line in help_lines:
                self._text(panel, line, (12, y), self.GREY, 0.45)
                y += self.LINE_HEIGHT
        return np.concatenate([img, panel], axis=1)


def define_skeleton_gui(bodypart_names: List[str], skeleton: dict = None) -> dict:  # pragma: no cover
    """Open a window to define a custom skeleton (see SkeletonEditor).

    Args:
        bodypart_names (list): body part names of the tracking tables.
        skeleton (dict): optional skeleton to start from.

    Returns:
        skeleton (dict): the defined skeleton, or None if the window was cancelled.
    """
    window_name = "deepOF - define skeleton"
    editor = SkeletonEditor(bodypart_names, skeleton)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(
        window_name, lambda event, x, y, flags, param: editor.handle_click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None
    )
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # Silently ignore if not supported
    result = None
    while result is None:
        cv2.imshow(window_name, editor.render())
        key = cv2.waitKey(20) & 0xFF
        if key != 255:
            result = editor.handle_key(key)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            result = "cancel"
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(window_name)
    return editor.to_skeleton() if result == "done" else None
