import tkinter as tk
from tkinter import ttk
import pandas as pd
import plotly.express as px
from io import BytesIO
from PIL import Image, ImageTk

# Load generated data
df = pd.read_csv('./data/synthetic_blood_flow_data.csv')

# Class for the Dashboard using Tkinter
class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthetic Blood Flow Dashboard")
        self.geometry("1000x800")

        # Create widgets for variable selection for the plots
        self.create_widgets()

    def create_widgets(self):
        # X and Y variable selection
        self.label_x = tk.Label(self, text="X-axis Variable:")
        self.label_x.pack(pady=10)
        self.xaxis_column = ttk.Combobox(self, values=list(df.columns))
        self.xaxis_column.current(0)
        self.xaxis_column.pack(pady=5)

        self.label_y = tk.Label(self, text="Y-axis Variable:")
        self.label_y.pack(pady=10)
        self.yaxis_column = ttk.Combobox(self, values=list(df.columns))
        self.yaxis_column.current(1)
        self.yaxis_column.pack(pady=5)

        # Button to generate plot
        self.generate_button = tk.Button(self, text="Generate Plot", command=self.update_graph)
        self.generate_button.pack(pady=10)

        # Frame to display the plot
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(pady=20)

        # Start with a default plot
        self.update_graph()

    def update_graph(self):
        xaxis_column = self.xaxis_column.get()
        yaxis_column = self.yaxis_column.get()

        # Create the plot with Plotly
        fig = px.scatter(df, x=xaxis_column, y=yaxis_column, trendline='ols', opacity=0.5)
        fig.update_layout(title=f'{yaxis_column} vs {xaxis_column}', xaxis_title=xaxis_column, yaxis_title=yaxis_column)

        # Convert the plot to an image and display it in Tkinter
        self.display_plot(fig)

    def display_plot(self, fig):
        # Remove previous plot if any
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Convert Plotly plot to image
        img_data = fig.to_image(format="png")
        image = Image.open(BytesIO(img_data))
        image = image.resize((800, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Create an image label and place it in the window
        img_label = tk.Label(self.graph_frame, image=photo)
        img_label.image = photo
        img_label.pack()

# Run the Tkinter application
if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
