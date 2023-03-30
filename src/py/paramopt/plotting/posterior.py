from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch


def plot_posterior(ax: plt.Axes, posterior, n_points: int = 300,
                   limits: Optional[np.array] = None, **kwargs
                   ) -> mpl.collections.QuadMesh:
    '''
    Plot the posterior probability in a heat map.

    :param ax: Axis in which to plot the posterior distribution.
    :param posterior: Posterior for which to plot the probability distribution.
        The parameters of the posterior are assumed to be two-dimensional.
    :param n_points: Number of sample points per axis.
    :param limits: Limits of the parameters which are plotted. This is needed
        if the `posterior` does not have a `prior` member from which the limits
        can be extracted.
    :returns: Artist created by :meth:`plt.Axes.pcolormesh`.
    '''

    if limits is None:
        if not hasattr(posterior, 'prior'):
            raise ValueError('Please supply limits for the parameters.')
        lower_limits = posterior.prior.support.base_constraint.lower_bound
        upper_limits = posterior.prior.support.base_constraint.upper_bound
    else:
        lower_limits = limits[:, 0]
        upper_limits = limits[:, 1]
    x_values = np.linspace(lower_limits[0], upper_limits[0], n_points)
    y_values = np.linspace(lower_limits[1], upper_limits[1], n_points)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    sample_points = torch.Tensor(np.array([x_mesh.flatten(),
                                           y_mesh.flatten()]).T)

    prop = np.exp(posterior.log_prob(sample_points).reshape(n_points, -1))

    default_kwargs = {'edgecolor': 'none', 'cmap': 'cividis'}
    default_kwargs.update(kwargs)
    return ax.pcolormesh(x_mesh, y_mesh, prop, **default_kwargs)


def label_low_high(colorbar: mpl.colorbar.Colorbar,
                   low: int = 0.1,
                   high: int = 0.9) -> None:
    '''
    Label the colorbar with 'low' and 'high' labels.

    :param colorbar: Colorbar for which to set the tick labels.
    :param low: Fraction along the colorbar where the 'low' label is placed.
    :param high: Fraction along the colorbar where the 'high' label is placed.
    '''
    v_low = colorbar.vmin + low * (colorbar.vmax - colorbar.vmin)
    v_high = colorbar.vmin + high * (colorbar.vmax - colorbar.vmin)
    colorbar.ax.set_yticks([v_low, v_high], ['low', 'high'],
                           rotation='vertical',
                           va='center')
    colorbar.ax.yaxis.set_tick_params(which='both', length=0)
