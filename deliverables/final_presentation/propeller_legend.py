import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_propeller_legend_plot(propeller_names, half_props=True):
    """
    Creates a standalone plot with a vertical list of dot legend entries on the left
    and the corresponding propeller images on the right. If any image is portrait
    (height > width), it will be rotated 90° to become landscape. If half_props=True,
    each image is cropped to its bottom half before display.
    
    Parameters:
    -----------
    propeller_names : list of str
        A list of propeller names. Each name corresponds to an image located at:
        'app/props/images/{propeller_name}.png'
    half_props : bool, default=True
        If True, crop each image to its bottom half (after any rotation).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the legend entries and (possibly rotated/cropped) images.
    """
    n = len(propeller_names)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n)]
    
    # Load all images first to calculate total height needed
    processed_images = []
    total_height = 0
    max_width = 0
    
    for name in propeller_names:
        img_path = f'app/props/images/{name}.png'
        try:
            img = Image.open(img_path)
            w, h = img.size
            
            # Rotate portrait to landscape
            if h > w:
                img = img.transpose(Image.ROTATE_270)
                w, h = img.size
            
            # Crop to bottom half if requested
            if half_props:
                img = img.crop((0, 0, w//2, h))
                w = w // 2
            
            processed_images.append((img, w, h))
            total_height += h
            max_width = max(max_width, w)
            
        except FileNotFoundError:
            # Use placeholder dimensions for missing images
            processed_images.append((None, 200, 50))
            total_height += 50
            max_width = max(max_width, 200)
    
    # Create figure with appropriate size
    fig_width = 10
    fig_height = max(8, total_height / 50)  # Scale based on total image height
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Calculate positions for each image
    current_y = 1.0
    y_positions = []
    
    for img_data in processed_images:
        _, w, h = img_data
        img_height_ratio = h / total_height
        y_positions.append((current_y - img_height_ratio, img_height_ratio))
        current_y -= img_height_ratio
    
    # Create subplots with variable heights
    for i, (name, img_data) in enumerate(zip(propeller_names, processed_images)):
        img, w, h = img_data
        y_bottom, height_ratio = y_positions[i]
        
        # Left column: dot + label
        ax_dot = fig.add_axes([0.05, y_bottom, 0.3, height_ratio])
        ax_dot.scatter([0.1], [0.5], color=colors[i], s=100)
        ax_dot.text(0.2, 0.5, name, fontsize=18, va='center', ha='left')
        ax_dot.set_xlim(0, 1)
        ax_dot.set_ylim(0, 1)
        ax_dot.axis('off')
        
        # Right column: image at original scale, right-aligned
        # Calculate right-aligned position within the image area
        img_area_width = 0.55
        img_x_pos = 0.4 + img_area_width - (w / max_width * img_area_width)
        img_width_ratio = w / max_width * img_area_width
        
        ax_img = fig.add_axes([img_x_pos, y_bottom, img_width_ratio, height_ratio])
        
        if img is not None:
            ax_img.imshow(img)
            ax_img.set_xlim(0, w)
            ax_img.set_ylim(h, 0)  # Flip y-axis for proper image orientation
        else:
            ax_img.text(
                0.5, 0.5,
                f"Image not found:\n{name}.png",
                ha='center', va='center', fontsize=12, color='red',
                transform=ax_img.transAxes
            )
        
        ax_img.axis('off')
    
    return fig


def create_multi_column_propeller_legend(propeller_names, n_columns=2, half_props=True):
    """
    Creates a multi-column propeller legend by splitting the propeller list into n columns.
    
    Parameters:
    -----------
    propeller_names : list of str
        A list of propeller names
    n_columns : int, default=2
        Number of columns to split the legend into
    half_props : bool, default=True
        If True, crop each image to its bottom half
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The combined figure with multiple columns
    """
    if n_columns == 1:
        return create_propeller_legend_plot(propeller_names, half_props)
    
    # Split propeller names into n_columns
    n_props = len(propeller_names)
    props_per_column = (n_props + n_columns - 1) // n_columns  # Ceiling division
    
    columns = []
    for i in range(n_columns):
        start_idx = i * props_per_column
        end_idx = min(start_idx + props_per_column, n_props)
        if start_idx < n_props:
            columns.append(propeller_names[start_idx:end_idx])
    
    # Process all images once to get dimensions
    all_processed_images = {}
    max_widths = []
    total_heights = []
    
    for col_idx, column_props in enumerate(columns):
        processed_images = []
        total_height = 0
        max_width = 0
        
        for name in column_props:
            img_path = f'app/props/images/{name}.png'
            try:
                img = Image.open(img_path)
                w, h = img.size
                
                # Rotate portrait to landscape
                if h > w:
                    img = img.transpose(Image.ROTATE_270)
                    w, h = img.size
                
                # Crop to bottom half if requested
                if half_props:
                    img = img.crop((w//2, 0, w, h))
                    w = w // 2
                
                processed_images.append((img, w, h))
                total_height += h
                max_width = max(max_width, w)
                
            except FileNotFoundError:
                processed_images.append((None, 200, 50))
                total_height += 50
                max_width = max(max_width, 200)
        
        all_processed_images[col_idx] = processed_images
        max_widths.append(max_width)
        total_heights.append(total_height)
    
    # Calculate combined figure dimensions
    total_fig_height = max(max(8, h / 50) for h in total_heights)
    total_fig_width = sum(10 for _ in columns)  # 10 units per column
    
    # Create combined figure
    fig = plt.figure(figsize=(total_fig_width, total_fig_height))
    
    # Color map
    cmap = plt.get_cmap('tab10')
    
    # Draw each column
    col_start_x = 0
    col_width = 1.0 / n_columns
    
    for col_idx, column_props in enumerate(columns):
        processed_images = all_processed_images[col_idx]
        max_width = max_widths[col_idx]
        total_height = total_heights[col_idx]
        
        # Calculate y positions for this column
        current_y = 1.0
        y_positions = []
        
        for img_data in processed_images:
            _, w, h = img_data
            img_height_ratio = h / total_height
            y_positions.append((current_y - img_height_ratio, img_height_ratio))
            current_y -= img_height_ratio
        
        # Draw each item in this column
        for i, (name, img_data) in enumerate(zip(column_props, processed_images)):
            img, w, h = img_data
            y_bottom, height_ratio = y_positions[i]
            
            # Get color (continue from previous columns)
            prop_idx = sum(len(columns[j]) for j in range(col_idx)) + i
            color = cmap(prop_idx % 10)
            
            # Left part: dot + label
            dot_left = col_start_x + 0.02
            dot_width = col_width * 0.3
            ax_dot = fig.add_axes([dot_left, y_bottom, dot_width, height_ratio])
            ax_dot.scatter([0.1], [0.5], color=color, s=400)  # Increased from 100 to 200
            ax_dot.text(0.2, 0.5, name, fontsize=28, va='center', ha='left')
            ax_dot.set_xlim(0, 1)
            ax_dot.set_ylim(0, 1)
            ax_dot.axis('off')
            
            # Right part: image (right-aligned within column)
            img_area_width = col_width * 0.65
            if img is not None:
                img_width_ratio = w / max_width * img_area_width
                img_x_pos = col_start_x + col_width * 0.35 + img_area_width - img_width_ratio
                
                ax_img = fig.add_axes([img_x_pos, y_bottom, img_width_ratio, height_ratio])
                ax_img.imshow(img)
                ax_img.set_xlim(0, w)
                ax_img.set_ylim(h, 0)
            else:
                img_x_pos = col_start_x + col_width * 0.35
                ax_img = fig.add_axes([img_x_pos, y_bottom, img_area_width, height_ratio])
                ax_img.text(
                    0.5, 0.5,
                    f"Image not found:\n{name}.png",
                    ha='center', va='center', fontsize=12, color='red', weight='bold',
                    transform=ax_img.transAxes
                )
            
            ax_img.axis('off')
        
        col_start_x += col_width
    
    return fig

if __name__ == "__main__":
    # Example usage
    propeller_names = [
        'dalprop5045',
        'dalprop6045',
        'dalprop4045',
        'printed5045'
    ]
    fig = create_multi_column_propeller_legend(propeller_names)
    plt.show()  # Display the plot
    # fig.savefig('propeller_legend.png', dpi=300, bbox_inches='tight')  # Save the figure if needed