import math
from IPython.display import display, HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepof.data import Coordinates

def radians_to_degrees(radians):
    return np.degrees(radians)


current_frame_index = 0
coords = None  
df = None  
sampled_coords = None 
start_frame = 0
end_frame = 0

def display_GUI(coordinates: Coordinates, experiment_id: str, center: str = "arena", align: str = None):
    global coords, df
    
    # Load coordinates
    coords = coordinates.get_coords_at_key(
        center=center,
        align=align,
        scale=coordinates._scales[experiment_id],
        key=experiment_id
    )

    def get_multi_select():
        if main_dropdown.value == "angle":
            plottingpoints = coordinates.get_angles()
        elif main_dropdown.value == "distance":
            plottingpoints = coordinates.get_distances()
        elif main_dropdown.value == "speed":
            plottingpoints = coordinates.get_coords(speed=1)
        global df
        df = plottingpoints[experiment_id]   
        
        columns = [(f"{col}", col) if isinstance(col, tuple) else (str(col), col) for col in df.columns]

        return columns

    # UI widgets
    main_dropdown = widgets.Dropdown(
        options=["angle", "distance", "speed"],
        value="distance",
        description="Select:",
        style={'description_width': 'initial'}
    )
    multiselect = widgets.SelectMultiple(
        options=get_multi_select(),
        value=[],
        description="Values:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px', height='200px', display='')
    )
    start_box = widgets.IntText(value=0, description="Start Frame:", layout=widgets.Layout(width='150px'))
    end_box = widgets.IntText(value=10, description="End Frame:", layout=widgets.Layout(width='150px'))
    range_selector = widgets.HBox([start_box, end_box])

    output = widgets.Output()
   
    def update_multiselect_visibility(change):
        if change['new'] in ["angle", "distance", "speed"]:
            multiselect.options = get_multi_select()
            multiselect.layout.display = ''

    main_dropdown.observe(update_multiselect_visibility, names='value')

    # Plotting function
    def plot_current_frame():
        global current_frame_index
        
        if sampled_coords is None or sampled_coords.empty:
            print("No frames to display. Please select a valid range.")
            return

        if not (0 <= current_frame_index < len(sampled_coords)):
            print(f"Invalid frame index: {current_frame_index}. Skipping plot.")
            return
        with output:
            output.clear_output()
            fig, ax = plt.subplots(figsize=(8, 8))
            selected_columns = list(multiselect.value)
            

            # Use global sampled_coords for the current frame
            
            frame_coords = sampled_coords.iloc[[current_frame_index]]  # Ensure DataFrame output
           
            if isinstance(frame_coords, pd.Series):
                x_coords = frame_coords['x']
                y_coords = frame_coords['y']
            else:
                x_coords = frame_coords.xs('x', level=1, axis=1)
                y_coords = frame_coords.xs('y', level=1, axis=1)

            # Set plot limits and invert y-axis
            ax.set_xlim(x_coords.min().min(), x_coords.max().max())
            ax.set_ylim(y_coords.min().min(), y_coords.max().max())
            ax.set_aspect('equal', adjustable='datalim')
            ax.invert_yaxis()

            # Plot points with labels
            for point_name in x_coords.columns:
                x, y = x_coords[point_name], y_coords[point_name]
                ax.scatter(x, y, color="blue", s=20)
                #ax.text(x, y, point_name, color="green", fontsize=8, ha="center", va="center")
            
            if main_dropdown.value == "angle" :
                filtered_df=df[selected_columns]
                for (point1, point2, point3), angle in filtered_df.iloc[[current_frame_index]].items():
                    
                    
                    # Check if all points exist in x_coords and y_coords
                    if point1 in x_coords.columns and point2 in x_coords.columns and point3 in x_coords.columns:
                        x1, y1 = x_coords.iloc[[0]][point1], y_coords.iloc[[0]][point1]
                        x2, y2 = x_coords.iloc[[0]][point2], y_coords.iloc[[0]][point2]
                        x3, y3 = x_coords.iloc[[0]][point3], y_coords.iloc[[0]][point3]

                        # Draw connections
                        ax.plot([x1, x2], [y1, y2], color="blue", linewidth=2.5)
                        ax.plot([x2, x3], [y2, y3], color="blue", linewidth=2.5)

                        # Calculate angle offset
                        dx1, dy1 = x1 - x2, y1 - y2
                        dx3, dy3 = x3 - x2, y3 - y2
                        offset_x = (dx1 + dx3) / 4
                        offset_y = (dy1 + dy3) / 4

                        # Display angle
                        #print(radians_to_degrees(angle))
                        if isinstance(angle, pd.Series):
                            scalar_angle = angle.iloc[0]  # Extract the scalar value
                        else:
                            scalar_angle = angle
                        ax.text(
                            x2 + offset_x, y2 + offset_y,
                            f"{radians_to_degrees(scalar_angle):.1f}°",
                            color="red", fontsize=8, ha="center"
                        )
                    else:
                        print(f"Skipping missing points for frame {current_frame_index}: {point1}, {point2}, {point3}")      
            if main_dropdown.value == "distance":
                filtered_df = df[selected_columns]
                for (point1, point2), distance in filtered_df.iloc[[current_frame_index]].items():
                    if point1 in x_coords.columns and point2 in x_coords.columns:
                        try:
                            x1, y1 = x_coords.iloc[[0]][point1], y_coords.iloc[[0]][point1]
                            x2, y2 = x_coords.iloc[[0]][point2], y_coords.iloc[[0]][point2]

                            # Draw line between the points
                            ax.plot([x1, x2], [y1, y2], color="green", linewidth=2.5)

                            # Display the distance at the midpoint
                            midpoint_x = (x1 + x2) / 2
                            midpoint_y = (y1 + y2) / 2
                            scalar_distance = distance.iloc[0] if isinstance(distance, pd.Series) else distance
                            ax.text(
                                midpoint_x, midpoint_y,
                                f"{scalar_distance:.2f}",
                                color="purple", fontsize=8, ha="center"
                            )
                        except KeyError as e:
                            print(f"KeyError: Missing data for points {point1}, {point2}: {e}")
                        except Exception as e:
                            print(f"Error while processing points {point1}, {point2}: {e}")
                    else:
                        print(f"Skipping missing points for frame {current_frame_index}: {point1}, {point2}")
            if main_dropdown.value == "speed":
                # Filter the DataFrame for the selected points
                filtered_df = df[selected_columns]

                for point, speed in filtered_df.iloc[[current_frame_index]].items():
                    #print(current_frame_index)
                    #print(filtered_df.iloc[[4]].items)
                    if point in x_coords.columns:
                        try:
                            # Safely access the coordinates of the point
                            x, y = x_coords.iloc[0][point], y_coords.iloc[0][point]

                            # Display the speed value at the point
                            scalar_speed = speed.iloc[0] if isinstance(speed, pd.Series) else speed
                            ax.scatter(x, y, color="blue", s=20)  # Plot the point
                            ax.text(
                                x, y,
                                f"{scalar_speed:.2f}",  # Display speed with 2 decimal places
                                color="purple", fontsize=8, ha="center"
                            )
                        except KeyError as e:
                            print(f"KeyError: Missing data for point {point}: {e}")
                        except Exception as e:
                            print(f"Error while processing point {point}: {e}")
                    else:
                        print(f"Skipping missing point for frame {current_frame_index}: {point}")

            
            plt.title(f"Frame {start_frame + current_frame_index}")
            plt.show()

    # Button handlers
    def on_animate_button_clicked(_):
        global current_frame_index, sampled_coords, start_frame, end_frame
        with output:
            output.clear_output()
            start_frame = start_box.value
            end_frame = end_box.value

            if start_frame < 0 or end_frame > len(coords) or start_frame >= end_frame:
                print("Invalid frame range. Please adjust the start and end frame values.")
                return

            sampled_coords = coords.iloc[start_frame:end_frame]  # Filter frames
            current_frame_index = 0  # Start at the first frame in the range
            
            plot_current_frame()
    
    def on_next_button_clicked(_):
        global current_frame_index
        #sampled_coords = coords.iloc[start_frame:end_frame] 
        if current_frame_index < len(sampled_coords) - 1:  # Prevent out-of-bounds
            current_frame_index += 1
            
            plot_current_frame()
        else:
            print("Already at the last frame.")

    def on_prev_button_clicked(_):
        global current_frame_index
        if current_frame_index > 0:  # Prevent out-of-bounds
            current_frame_index -= 1
            plot_current_frame()
        else:
            print("Already at the first frame.")



    
    # Buttons
    animate_button = widgets.Button(description="Animate", button_style="")
    next_button = widgets.Button(description="Next", button_style="")
    prev_button = widgets.Button(description="Previous", button_style="")
    controls = widgets.VBox([
    widgets.HBox([main_dropdown, multiselect]),
    range_selector,
    widgets.HBox([prev_button,animate_button, next_button ]),
    ])

    animate_button.on_click(on_animate_button_clicked)
    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)

    # Display UI
    display(controls, output)
