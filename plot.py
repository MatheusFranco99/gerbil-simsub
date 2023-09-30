"""Plot functions"""

from typing import List
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.patches as patches
import numpy as np

plt.style.use('ggplot')

def my_hash(text:str) -> int:
    """Custom hash function for deterministic hashing."""
    hash_v=0
    for ch in text:
        hash_v = ( hash_v*281  ^ ord(ch)*997) & 0xFFFFFFFF
    return hash_v

def sanitize_output_f(output_f: str) -> str:
    """Sanitizes filename"""
    return "images/" + output_f.replace("/","_div_")

def plot_chart(x: List[float], y: List[float],
               x_label: str, y_label: str,
               error = None,
               ax = None,
               filename: str = None,
               savefig: bool = True, showfig: bool = False) -> None:
    """Creates a line plot"""

    if ax == None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, linestyle = "--", marker="o")

    if error is not None:
        _, caps, _ = ax.errorbar(x, y, yerr=error, color='r', linestyle='', capsize=5, alpha = 0.6)
        for cap in caps:
            cap.set_markeredgewidth(1)


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(f"{y_label} by {x_label}")
    ax.ticklabel_format(useOffset=False)
    if savefig:
        # plt.tight_layout()
        if filename == None:
            filename = f"line_chart_{x_label}_{y_label}_{my_hash(str((x,y)))}.png"
        plt.savefig(sanitize_output_f(filename))
    if showfig:
        plt.show()

def scatter_annotate_x(x: List[float], y: List[float], z: List[float],
                       y_label: str, z_label: str,
                       y_error = None, z_error = None,
                       sections: bool = False, num_colors: int = 80, # flag to generate {num_colors} eclipses as a heat map
                       savefig: bool = True, showfig: bool = False) -> None:
    """Create a 2D scatter plot of points (y, z) and annotate each point with the x value."""
    plt.figure(figsize=(15,8))


    if sections:
        # Main color steps for gradient
        colors = [
            (1, 1, 1),    # White
            (1, 0, 0),    # Red
            (1, 1, 0),    # Yellow
            (0, 1, 0),    # Green
            (0.5, 0, 0.5) # Purple
        ]

        # Interpolate between the defined colors
        gradient_colors = []
        for i in range(len(colors) - 1):
            for j in range(num_colors // (len(colors) - 1)):
                r = np.interp(j, [0, num_colors // (len(colors) - 1)], [colors[i][0], colors[i + 1][0]])
                g = np.interp(j, [0, num_colors // (len(colors) - 1)], [colors[i][1], colors[i + 1][1]])
                b = np.interp(j, [0, num_colors // (len(colors) - 1)], [colors[i][2], colors[i + 1][2]])
                gradient_colors.append((r, g, b))

        # Convert RGB colors to hexadecimal
        hex_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in gradient_colors]

        vertical_radius = 1.5*max(z)/num_colors
        horizontal_radius = 1.5*max(y)/num_colors

        for i in range(num_colors-1,0,-1):
            ellipse = patches.Ellipse((0, 0), 2 * horizontal_radius*i, 2 * vertical_radius*i,
                                    fill=True, color=hex_colors[i], linestyle='dashed', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.axline((0, 0), slope=vertical_radius / horizontal_radius, color='gray', linestyle='--')

        coefficients = np.polyfit(y, z, 2)
        polynomial = np.poly1d(coefficients)
        x_values = np.linspace(0, max(y) + 0.2 *(max(y) - min(y)), 1000)  # Generating 1000 points for a smooth curve
        y_values = polynomial(x_values)
        plt.plot(x_values, y_values, color='grey', alpha=0.7, linestyle='--')  # Fitted curve


    # Create a scatter plot of (y, z)
    plt.scatter(y, z, marker='o', label='(y, z) Points',color='blue')


    if y_error is not None:
        _, caps, _ = plt.errorbar(y, z, xerr=y_error, color='b', linestyle='', capsize=5)
        for cap in caps:
            cap.set_markeredgewidth(1)
    if z_error is not None:
        _, caps, _ = plt.errorbar(y, z, yerr=z_error, color='b', linestyle='', capsize=5)
        for cap in caps:
            cap.set_markeredgewidth(1)

    # Annotate each point with the x value
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        plt.annotate(f'{str(xi)}', (yi, zi), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} by {y_label}")

    plt.ylim(bottom=max(0,min(z)-(max(z)-min(z))/10),top=max(z)+(max(z)-min(z))/10)
    plt.xlim(left=max(0,min(y)-(max(y)-min(y))/10),right=max(y)+(max(y)-min(y))/10)

    plt.grid(True)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(sanitize_output_f(f"annotated_{y_label}_{z_label}_{my_hash(str((x,y,z)))}.png"))
    if showfig:
        plt.show()

def plot_3d_curve(x: List[float], y: List[float], z: List[float],
                  x_label: str, y_label: str, z_label: str,
                  x_error = None, y_error = None, z_error = None,
                  annotation: List | None = None,
                  savefig: bool = False, showfig: bool = False) -> None:
    """Create 3D scatter plot with curve and annotation flag"""

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(x, y, z, c='b', marker='o', label='Data Points')

    # Define a grid for interpolation
    xi, yi = np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data to generate the curve
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Plot the interpolated curve using plot_surface
    ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if annotation != None:
        # Annotate each point with the x value
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.text(xi,yi,zi,f'{str(annotation[i])}',None)

    if x_error is not None or y_error is not None or z_error is not None:
        _, caps, _ = ax.errorbar(x,y,z, xerr = x_error, yerr = y_error, zerr = z_error, color='b', linestyle='', capsize=5)
        for cap in caps:
            cap.set_markeredgewidth(1)


    plt.tight_layout()

    if savefig:
        plt.savefig(sanitize_output_f(f"3d_curve_{x_label}_{y_label}_{z_label}_{my_hash(str((x,y,z)))}.png"))
    if showfig:
        plt.show()
