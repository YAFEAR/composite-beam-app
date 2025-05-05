# cross_section.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cross_section(case_type, h, b, t_flange, t_web, t_side, t_core_web=5, t_core_side=5):
    """
    Draw the cross-section of the beam.
    
    Parameters:
    - case_type: int (1=open I-beam, 2=semi-closed I-beam, 3=full box)
    - h: height [mm]
    - b: width [mm]
    - t_flange: flange thickness [mm]
    - t_web: carbon web wall thickness [mm]
    - t_side: carbon side wall thickness (used in case 3)
    - t_core_web: width of foam in central web
    - t_core_side: width of side foam pieces (used in case 2)
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(-5, b + 5)
    ax.set_ylim(-5, h + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Carbon top and bottom flanges
    ax.add_patch(patches.Rectangle((0, h - t_flange), b, t_flange, color="black"))
    ax.add_patch(patches.Rectangle((0, 0), b, t_flange, color="black"))

    if case_type == 1:
        # Carbon web walls
        web_left = (b - t_core_web) / 2 - t_web
        web_right = (b + t_core_web) / 2
        ax.add_patch(patches.Rectangle((web_left, t_flange), t_web, h - 2*t_flange, color="black"))
        ax.add_patch(patches.Rectangle((web_right, t_flange), t_web, h - 2*t_flange, color="black"))
        # Foam in between web walls
        ax.add_patch(patches.Rectangle(((b - t_core_web)/2, t_flange), t_core_web, h - 2*t_flange, color="orange", alpha=0.6))

    elif case_type == 2:
        # Carbon web walls
        web_left = (b - t_core_web) / 2 - t_web
        web_right = (b + t_core_web) / 2
        ax.add_patch(patches.Rectangle((web_left, t_flange), t_web, h - 2*t_flange, color="black"))
        ax.add_patch(patches.Rectangle((web_right, t_flange), t_web, h - 2*t_flange, color="black"))
        # Foam in web
        ax.add_patch(patches.Rectangle(((b - t_core_web)/2, t_flange), t_core_web, h - 2*t_flange, color="orange", alpha=0.5))
        # Foam side closures
        foam_side_width = (b - t_core_web - 2 * t_web) / 2
        ax.add_patch(patches.Rectangle((0, t_flange), foam_side_width, h - 2*t_flange, color="orange", alpha=0.4))
        ax.add_patch(patches.Rectangle((b - foam_side_width, t_flange), foam_side_width, h - 2*t_flange, color="orange", alpha=0.4))
   
    elif case_type == 3:
        # Carbon side walls
        ax.add_patch(patches.Rectangle((0, t_flange), t_side, h - 2*t_flange, color="black"))
        ax.add_patch(patches.Rectangle((b - t_side, t_flange), t_side, h - 2*t_flange, color="black"))
        # Foam core
        ax.add_patch(patches.Rectangle((t_side, t_flange), b - 2*t_side, h - 2*t_flange, color="orange", alpha=0.7))

    else:
        raise ValueError("Invalid case_type. Choose 1, 2, or 3.")

    title_map = {
        1: "Case 1: Open I-beam",
        2: "Case 2: Semi-Closed I-beam",
        3: "Case 3: Full Box Beam"
    }
    ax.set_title(title_map[case_type], fontsize=14)
    plt.show()