import pandas as pd
import matplotlib.pyplot as plt

# XGB Test Report 数据
data1 = [
    {'class': 0, 'precision': 0.92, 'recall': 0.97, 'f1-score': 0.95, 'support': 873},
    {'class': 1, 'precision': 0.97, 'recall': 0.94, 'f1-score': 0.96, 'support': 858},
    {'class': 2, 'precision': 0.67, 'recall': 0.55, 'f1-score': 0.60, 'support': 877},
    {'class': 3, 'precision': 0.92, 'recall': 0.90, 'f1-score': 0.91, 'support': 914},
    {'class': 4, 'precision': 0.69, 'recall': 0.73, 'f1-score': 0.71, 'support': 737},
    {'class': 5, 'precision': 0.76, 'recall': 0.73, 'f1-score': 0.75, 'support': 1072},
    {'class': 6, 'precision': 0.60, 'recall': 0.53, 'f1-score': 0.56, 'support': 682},
    {'class': 7, 'precision': 0.57, 'recall': 0.60, 'f1-score': 0.58, 'support': 813},
    {'class': 8, 'precision': 0.70, 'recall': 0.83, 'f1-score': 0.76, 'support': 1174},
    {'class': 'accuracy', 'precision': 0.76, 'recall': '-', 'f1-score': '-', 'support': 8000},
    {'class': 'macro avg', 'precision': 0.76, 'recall': 0.75, 'f1-score': 0.75, 'support': 8000},
    {'class': 'weighted avg', 'precision': 0.76, 'recall': 0.76, 'f1-score': 0.76, 'support': 8000}
]

# XGB Test Report (On denoise data) 数据
data2 = [
    {'class': 0, 'precision': 0.91, 'recall': 0.97, 'f1-score': 0.94, 'support': 873},
    {'class': 1, 'precision': 0.98, 'recall': 0.92, 'f1-score': 0.95, 'support': 858},
    {'class': 2, 'precision': 0.69, 'recall': 0.33, 'f1-score': 0.45, 'support': 877},
    {'class': 3, 'precision': 0.81, 'recall': 0.90, 'f1-score': 0.85, 'support': 914},
    {'class': 4, 'precision': 0.72, 'recall': 0.65, 'f1-score': 0.68, 'support': 737},
    {'class': 5, 'precision': 0.69, 'recall': 0.75, 'f1-score': 0.72, 'support': 1072},
    {'class': 6, 'precision': 0.43, 'recall': 0.67, 'f1-score': 0.53, 'support': 682},
    {'class': 7, 'precision': 0.56, 'recall': 0.54, 'f1-score': 0.55, 'support': 813},
    {'class': 8, 'precision': 0.72, 'recall': 0.70, 'f1-score': 0.71, 'support': 1174},
    {'class': 'accuracy', 'precision': 0.72, 'recall': '-', 'f1-score': '-', 'support': 8000},
    {'class': 'macro avg', 'precision': 0.72, 'recall': 0.72, 'f1-score': 0.71, 'support': 8000},
    {'class': 'weighted avg', 'precision': 0.73, 'recall': 0.72, 'f1-score': 0.71, 'support': 8000}
]

# Convert to DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Function to plot table
def plot_table(df, title, image_path):
    # Plot table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Adjust font size and style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Bold the header and set font color to white
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')  # Set header font color to white
            cell.set_facecolor('#2f5697')  # Deep blue background for the header
        else:
            cell.set_facecolor('#f9f9f9')  # Light background for other cells
        cell.set_edgecolor('black')  # Black borders for cells

    # Set the title
    plt.title(title, fontsize=14)

    # Make layout tight
    plt.tight_layout()

    # Save image
    plt.savefig(image_path, dpi=300)

# Generate first table (XGB Test Report)
plot_table(df1, "XGB Test Report", "xgb_test_report.png")

# Generate second table (XGB Test Report On Denoise Data)
plot_table(df2, "XGB Test Report (On denoise data)", "xgb_test_report_denoise.png")