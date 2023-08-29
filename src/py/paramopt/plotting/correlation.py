'''
Plot correlation matrices.
'''

from typing import List, Optional, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_correlation_matrix(ax: plt.Axes, data: pd.DataFrame,
                            parameter_names: Optional[List[str]] = None
                            ) -> mpl.cm.ScalarMappable:
    '''
    Correlation matrix of the given data.

    :param ax: Axes in which to plot the correlation matrix.
    :param data: For the columns of this data frame the correlation is
        calculated and plotted.
    :param parameter_names: If given, set x- and y-labels.
    :return: ScalarMappable for colorbar.
    '''
    corr = data.corr()
    mappable = circled_correlation_matrix(ax, corr.values, cmap='coolwarm')

    # Disable tick labels per default, only set when parameter_names are given
    ax.tick_params(which='both', left=False, labelleft=False,
                   bottom=False, labelbottom=False)
    if parameter_names is not None:
        set_xlabels(ax, parameter_names)
        set_ylabels(ax, parameter_names)

    ax.spines[:].set_visible(False)

    return mappable


def circled_correlation_matrix(
        ax: plt.Axes, data: np.ndarray,
        cmap: Optional[Union[mpl.colors.Colormap, str]] = None
) -> mpl.cm.ScalarMappable:
    '''
    Plot correlation matrix with circles for each correlation.

    The color of the circle encodes the correlation. The area of the circle is
    scaled by the absolute correlation.

    :param ax: Axes in which to plot the correlation matrix.
    :param data: Correlation matrix.
    :param cmap: Colormap to use.
    :return: ScalarMappable for colorbar.
    '''
    if data.min() < -1 or data.max() > 1:
        raise ValueError("Only values from -1 to 1 are allowed")

    cmap = cmap or mpl.rcParams["image.cmap"]
    if isinstance(cmap, str):
        cmap = plt.colormaps.get(cmap)

    def get_color(value: float):
        return cmap((value + 1) / 2)

    # use 0.9 instead of 1 to leave some space between circles
    max_radius = 0.9 / np.max(data.shape) / 2

    def get_radius(value: float):
        return np.sqrt(np.abs(value)) * max_radius

    y_step = 1 / data.shape[0]
    x_step = 1 / data.shape[1]

    for row, col in np.ndindex(data.shape):
        val = data[row, col]
        circle = mpl.patches.Circle(
            ((0.5 + col) * x_step, 1 - (0.5 + row) * y_step),
            radius=get_radius(val),
            color=get_color(val),
            transform=ax.transAxes)
        ax.add_artist(circle)

    # correct limits; this also inverts y-axis
    ax.set_ylim(data.shape[0] - 0.5, -0.5)
    ax.set_xlim(-0.5, data.shape[1] - 0.5)

    return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, 1), cmap=cmap)


def set_xlabels(ax: plt.Axes, parameter_names: List[str], rotation: int = -45
                ) -> None:
    '''
    Set the x-ticks labels of a correlation matrix.

    Place labels at the top of the matrix, rotate labels and hide ticks.

    :param ax: Axes in which to place the labels.
    :param rotation: Rotation of the labels.
    :param parameter_names: Parameter names.
    '''
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(np.arange(len(parameter_names)), labels=parameter_names)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right",
             rotation_mode="anchor")


def set_ylabels(ax: plt.Axes, parameter_names: List[str]) -> None:
    '''
    Set the y-ticks labels of a correlation matrix.

    Place labels at the left of the matrix and hide ticks.

    :param ax: Axes in which to place the labels.
    :param parameter_names: Parameter names.
    '''
    ax.tick_params(axis='y', length=0)
    ax.set_yticks(np.arange(len(parameter_names)), labels=parameter_names)
    ax.tick_params(left=False, labelleft=True)
