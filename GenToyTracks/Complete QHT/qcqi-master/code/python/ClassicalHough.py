"""Classical Hough transform."""
import numpy as np
from math import sqrt


def HoughLine(x_coords, y_coords, theta_step=1, width=1000, height=1000):
    """Hough transform for lines.

    Args:
        width: Width, in arbitrary units, of input.
        height: Height, in arbitrary units, of input.
        theta_step: Angle, in degrees, to step/increment theta.
        x_coords: List of x-coordinates of hits.
        y_coords: List of y-coordinates of hits.

    Returns:
        accumulator: 2D array of the Hough transform accumulator.
        thetas: Array of angles, in radians, used for computation of rho.
        rhos: Array of rho values. Max size is 2x diagonal of input.
    """

    _max_length = max(max(x_coords), max(y_coords)) + 1
    _thetas = np.linspace(0, np.pi, _max_length, endpoint=True)
    _rhos = np.linspace(0, _max_length, _max_length)

    # Cached variables
    _num_thetas = len(_thetas)
    _num_rhos = len(_rhos)
    _cos_t = np.cos(_thetas)
    _sin_t = np.sin(_thetas)

    # Accumulator array of theta vs. rho
    _accumulator = np.zeros((_num_rhos, _num_thetas))

    # Vote in the accumulator
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]

        for itheta in range(_num_thetas):
            # Calculate rho; add offset _max_length to ensure positive
            # values only.
            rho = abs(int(round(x * _cos_t[itheta] + y * _sin_t[itheta])))
            _accumulator[rho, itheta] += 1
            # if (_accumulator[rho, itheta] > 5):
            print('(x,y): (%f, %f); acc: %f; rho: %d; theta: %f' %
                  (x, y, _accumulator[rho, itheta], rho, _thetas[itheta]))

    return _accumulator, _thetas, _rhos


def ShowHoughAccumulator(x_coords,
                         y_coords,
                         accumulator,
                         thetas,
                         rhos,
                         save_path=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(accumulator,
               cmap='Blues_r',
               extent=[thetas[0], thetas[-1], rhos[0], rhos[-1]],
               aspect='auto')
    ax.set_title('Hough transform')
    ax.set_xlabel(r'$\theta$ [rad]')
    ax.set_ylabel(r'$\rho$ [arb. units]')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def _calc_rho_from_phi(x_coord, y_coord, phi):
    """Calculates rho = x * cos(phi) + y * sin(phi).

    Note that the magnitude is taken to get the length since rho will be
    negative for 

    Args:
        x_point: x-coordinate of hit
        y_point: y-coordinate of hit
        phi: Angle between rho and the x-axis
    """
    rho = x_coord * np.cos(phi) + y_coord * np.sin(phi)
    # print("x: ", x_coord, ", y: ", y_coord, ", rho: ", rho, ", phi: ", phi)
    return abs(rho)


def plot_acc_matrix(acc_rho, acc_phi, num_binsx, num_binsy, acc_range):
    """Plots accumulator matrix, i.e. Hough space.

    Args:
        ht_rhos: List of rho values from Hough transform
        ht_phis: List of phi values ...
        num_binsx: Number of bins in x-coord (phi)
        num_binsy: Number of bins in y-coord (r)
        acc_range: Range of plot
    """
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(10, 8), dpi=90)
    histo = plt.hist2d(acc_phi,
                       acc_rho,
                       bins=(num_binsx, num_binsy),
                       cmap='jet',
                       range=acc_range)
    plt.colorbar(histo[3])
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\rho$ [mm]')
    plt.title('Accumulator matrix')
    plt.savefig('accum_matrix.png')
    plt.show()


def ConformalMapping(x_coords, y_coords):
    """Maps (x,y)-coordinates to (u,v)-coordinates.
    
    Args:
        x_coords: list of x-coordinates of hits
        y_coords: list of y-coordinates of hits

    Returns:
        Lists u- and v-coordinates, and r^2 = x^2 + y^2
    """
    r_squared = np.zeros(len(x_coords))
    u_coords = np.zeros(len(x_coords))
    v_coords = np.zeros(len(x_coords))

    for i in range(len(x_coords)):
        r_squared[i] = x_coords[i]**2 + y_coords[i]**2
        if r_squared[i] != 0:
            u_coords[i] = x_coords[i] / r_squared[i]
            v_coords[i] = y_coords[i] / r_squared[i]
        else:
            u_coords[i] = 0
            v_coords[i] = 0
            r_squared[i] = 0
    return r_squared, u_coords, v_coords


def ShowConformalMap(u_coords, v_coords):
    """Plots conformal map space.

    Args:
        u_coords: List of u-coordinates
        v_coords: List of v-coordinates
    """
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(10, 10), dpi=90)
    plt.scatter(u_coords, v_coords, s=10)
    plt.grid(b=True,
             which='major',
             axis='both',
             color='blue',
             linestyle='-',
             linewidth=0.1)
    plt.title('Conformal Map\n(u, v)-space')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.savefig('conformal_map.png')
    plt.show()


def plot_polar_coords(rhos, phis):
    import matplotlib.pyplot as plt

    ax = plt.axes(polar=True)

    width = 0.05
    bars = plt.bar(phis, rhos, width=0.01, bottom=width)

    for r, bar in zip(rhos, bars):
        bar.set_facecolor(plt.cm.jet(r / 20.))
        bar.set_alpha(1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()