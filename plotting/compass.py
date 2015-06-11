"""
This module provides a compass class for adding compass
indicators to matplotlib plots.
"""


import scipy as sp
import scipy.special as special
import matplotlib.patches as patches


class Compass():
    """
    This represents a compass that can be added to matplotlib plots.

    The compass consists of two arrows connected at their base in a
    right angle, indicating the directions of North and East.

    Attributes:
    north - float
        The angle in degrees from the positive x-axis to North.
    east - float
        The angle in degrees from the positive x-axis to East.
    """

    def __init__(self, north=90.0, east=0.0):
        """
        The default compass aligns N and E with the plot xy axes.

        Args:
        north - float, default=90
            The angle in degrees from the positive x-axis to North.
        east - float, default=0
            The angle in degrees from the positive x-axis to East.
        """
        self.north = north % 360
        self.east = east % 360

    def rotate(self, angle):
        """
        Rotate the compass with respect to the plot xy axes.

        Args:
        angle - float
            Counterclockwise angle in degrees by which to rotate the
            compass, keeping the xy plot axes fixed.
        """
        self.north = (self.north + angle) % 360
        self.east = (self.east + angle) % 360

    def reflect(self, axis):
        """
        Reflect the compass directions across a line.

        Args:
        axis - float
            The angle, measured in degrees counterclockwise from the
            positive x-axis of the xy plot axes, of a direction over
            which to reflect the compass directions.
        """
        self.north = (self.north + 2*(axis - self.north)) % 360
        self.east = (self.east + 2*(axis - self.east)) % 360

    def plotCompass(self, ax, position=(0,0), length=1.0,
                    color='k', width=0.1, head_width=4.0,
                    N_letter_scale=1.5, E_letter_scale=1.5,
                    weight='bold', size=14):
        start = sp.array(position)
        delta_north = sp.array([length*special.cosdg(self.north),
                                length*special.sindg(self.north)])
        delta_east = sp.array([length*special.cosdg(self.east),
                               length*special.sindg(self.east)])
        ax.patches.Arrow(start[0], start[1],
                 delta_north[0], delta_north[1],
                 width=width, color=color,
                 shape='full', head_width=head_width)
        ax.patches.Arrow(start[0], start[1],
                 delta_east[0], delta_east[1],
                 width=width, color=color,
                 shape='full', head_width=head_width)
        north_text_position = start + N_letter_scale*delta_north
        east_text_position = start + E_letter_scale*delta_east
        if size > 0:
            ax.text(north_text_position[0], north_text_position[1],
                    "N", color=color, weight=weight, size=size)
            ax.text(east_text_position[0], east_text_position[1],
                    "E", color=color, weight=weight, size=size)