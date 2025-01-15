import math
from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepof.data import Coordinates

class GUI:
    def __init__(self, coordinates: Coordinates, experiment_id: str, center: str = "arena", align: str = None):
        # Initialize data
        self.coordinates = coordinates
        self.experiment_id = experiment_id
        self.center = center
        self.align = align
        self.coords = coordinates.get_coords_at_key(
            center=center,
            align=align,
            scale=coordinates._scales[experiment_id],
            key=experiment_id
        )
        self.df = None
        self.sampled_coords = None
        self.current_frame_index = 0
        self.start_frame = 0
        self.end_frame = 0

        # Initialize widgets
        self.main_dropdown = widgets.Dropdown(
            options=["angle", "distance", "speed"],
            value="distance",
            description="Select:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        self.multiselect = widgets.SelectMultiple(
            options=[],
            value=[],
            description="Values:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='200px')
        )

        self.start_box = widgets.IntText(value=0, description="Start Frame:", layout=widgets.Layout(width='150px'))
        self.end_box = widgets.IntText(value=10, description="End Frame:", layout=widgets.Layout(width='150px'))

        self.range_selector = widgets.HBox(
            [self.start_box, self.end_box],
            layout=widgets.Layout(justify_content='space-between', width='400px')
        )

        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=10, step=1,
            description="Frame:", continuous_update=True,
            layout=widgets.Layout(width='500px')
        )

        self.animate_button = widgets.Button(description="Animate", button_style="", layout=widgets.Layout(width='150px'))
        self.output = widgets.Output()

        self.initialize_gui()

    def get_multi_select(self):
        if self.main_dropdown.value == "angle":
            plottingpoints = self.coordinates.get_angles()
        elif self.main_dropdown.value == "distance":
            plottingpoints = self.coordinates.get_distances()
        elif self.main_dropdown.value == "speed":
            plottingpoints = self.coordinates.get_coords(speed=1)
        
        self.df = plottingpoints[self.experiment_id]
        return [(str(col), col) if isinstance(col, tuple) else (str(col), col) for col in self.df.columns]

    def update_multiselect_visibility(self, change):
        if change['new'] in ["angle", "distance", "speed"]:
            self.multiselect.options = self.get_multi_select()

    def plot_current_frame(self, change=None):
        if self.sampled_coords is None or self.sampled_coords.empty:
            return

        frame_coords = self.sampled_coords.iloc[[self.current_frame_index]]

        with self.output:
            self.output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            selected_columns = list(self.multiselect.value)

            x_coords = frame_coords.xs('x', level=1, axis=1)
            y_coords = frame_coords.xs('y', level=1, axis=1)

            ax.set_xlim(x_coords.min().min(), x_coords.max().max())
            ax.set_ylim(y_coords.min().min(), y_coords.max().max())
            ax.set_aspect('equal', adjustable='datalim')
            ax.invert_yaxis()

            for point_name in x_coords.columns:
                x, y = x_coords[point_name], y_coords[point_name]
                ax.scatter(x, y, color="blue", s=20)

            if self.main_dropdown.value == "angle":
                self.plot_angles(ax, x_coords, y_coords, selected_columns)
            elif self.main_dropdown.value == "distance":
                self.plot_distances(ax, x_coords, y_coords, selected_columns)
            elif self.main_dropdown.value == "speed":
                self.plot_speeds(ax, x_coords, y_coords, selected_columns)

            plt.title(f"Frame {self.start_frame + self.current_frame_index}")
            plt.show()

    def plot_angles(self, ax, x_coords, y_coords, selected_columns):
        filtered_df = self.df[selected_columns]
        for (point1, point2, point3), angle in filtered_df.iloc[[self.current_frame_index]].items():
            if point1 in x_coords.columns and point2 in x_coords.columns and point3 in x_coords.columns:
                x1, y1 = x_coords.iloc[[0]][point1], y_coords.iloc[[0]][point1]
                x2, y2 = x_coords.iloc[[0]][point2], y_coords.iloc[[0]][point2]
                x3, y3 = x_coords.iloc[[0]][point3], y_coords.iloc[[0]][point3]

                ax.plot([x1, x2], [y1, y2], color="blue", linewidth=2.5)
                ax.plot([x2, x3], [y2, y3], color="blue", linewidth=2.5)

                dx1, dy1 = x1 - x2, y1 - y2
                dx3, dy3 = x3 - x2, y3 - y2
                offset_x = (dx1 + dx3) / 4
                offset_y = (dy1 + dy3) / 4

                scalar_angle = angle.iloc[0] if isinstance(angle, pd.Series) else angle
                ax.text(
                    x2 + offset_x, y2 + offset_y,
                    f"{np.degrees(scalar_angle):.1f}°",
                    color="red", fontsize=8, ha="center"
                )

    def plot_distances(self, ax, x_coords, y_coords, selected_columns):
        filtered_df = self.df[selected_columns]
        for (point1, point2), distance in filtered_df.iloc[[self.current_frame_index]].items():
            if point1 in x_coords.columns and point2 in x_coords.columns:
                x1, y1 = x_coords.iloc[[0]][point1], y_coords.iloc[[0]][point1]
                x2, y2 = x_coords.iloc[[0]][point2], y_coords.iloc[[0]][point2]

                ax.plot([x1, x2], [y1, y2], color="green", linewidth=2.5)
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2

                scalar_distance = distance.iloc[0] if isinstance(distance, pd.Series) else distance
                ax.text(
                    midpoint_x, midpoint_y,
                    f"{scalar_distance:.2f}",
                    color="purple", fontsize=8, ha="center"
                )

    def plot_speeds(self, ax, x_coords, y_coords, selected_columns):
        filtered_df = self.df[selected_columns]
        for point, speed in filtered_df.iloc[[self.current_frame_index]].items():
            if point in x_coords.columns:
                x, y = x_coords.iloc[0][point], y_coords.iloc[0][point]
                scalar_speed = speed.iloc[0] if isinstance(speed, pd.Series) else speed
                ax.scatter(x, y, color="blue", s=20)
                ax.text(
                    x, y,
                    f"{scalar_speed:.2f}",
                    color="purple", fontsize=8, ha="center"
                )

    def on_animate_button_clicked(self, _):
        self.start_frame = self.start_box.value
        self.end_frame = self.end_box.value

        if self.start_frame < 0 or self.end_frame > len(self.coords) or self.start_frame >= self.end_frame:
            with self.output:
                self.output.clear_output()
                print("Invalid frame range. Please adjust the start and end frame values.")
            return

        self.sampled_coords = self.coords.iloc[self.start_frame:self.end_frame]
        self.current_frame_index = 0
        self.frame_slider.min = 0
        self.frame_slider.max = len(self.sampled_coords) - 1
        self.frame_slider.value = 0
        self.plot_current_frame()
        
    def update_current_frame_index(self, change):
        self.current_frame_index = change['new']
        self.plot_current_frame()


    def initialize_gui(self):
        self.multiselect.options = self.get_multi_select()
        self.main_dropdown.observe(self.update_multiselect_visibility, names='value')
        self.frame_slider.observe(lambda change: self.update_current_frame_index(change), names='value')
        self.animate_button.on_click(self.on_animate_button_clicked)

        controls = widgets.VBox([
            widgets.HBox([
                self.main_dropdown, self.multiselect
            ], layout=widgets.Layout(justify_content='space-between', align_items='center')),
            self.range_selector,
            widgets.HBox([
                self.animate_button
            ], layout=widgets.Layout(justify_content='flex-start', align_items='center', width='500px')),
            self.frame_slider
        ], layout=widgets.Layout(spacing="20px"))

        display(controls, self.output)
