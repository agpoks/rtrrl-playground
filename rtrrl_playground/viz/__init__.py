"""Animations. A trajectory plot says where the car went; an animation says why.

Everything here writes a GIF through matplotlib's Pillow writer, which is the
only writer that needs no system binary -- no ffmpeg, no ImageMagick. GIFs also
render on Read the Docs with a plain ``{image}`` directive and no player, which
a video does not.
"""

from rtrrl_playground.viz.animate import (
    animate_episode, animate_learning, animate_safety, save_gif,
)

__all__ = ["animate_episode", "animate_learning", "animate_safety", "save_gif"]
