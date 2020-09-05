"""Simple detector class."""
import matplotlib.pyplot as plot
import ClassicalHough


class SimpleDetector:
    """Construct a simple detector.

    A simple detector is one which consists of concentric circles/layers
    of various radii. The detector can be plotted.
    
    Attributes:
        num_layers: Number of concentric circles making detector.
        layer_radii: Array of layer radii.
        layers: Concentric circles comprising the detector.
        num_hits: Number of hits on layers.
        hit_xy: 2D (x, y) hit locations.
    """
    def __init__(self,
                 num_layers=10,
                 layer_radii=range(100, 1100, 100),
                 num_hits=0,
                 hits_xy=None):
        """Init the class."""
        self.num_layers = num_layers
        self.layer_radii = layer_radii
        self.layers = []
        self.num_hits = 0
        self.hits_x = []
        self.hits_y = []
        self.construct()

    def construct(self):
        """Constructs the detector as a plot."""
        for layer in range(self.num_layers):
            self.layers.append(
                plot.Circle((0, 0),
                            self.layer_radii[layer],
                            color="gray",
                            linewidth=0.1,
                            fill=False))

    def plot_detector(self):
        """Plot the detector."""
        _fig, ax = plot.subplots(num=None, figsize=(10, 10), dpi=90)
        _max_extent = max(self.layer_radii)
        _fig_extents = (-((0.1 * _max_extent) + _max_extent),
                        (0.1 * _max_extent) + _max_extent)
        ax.set_xlim(_fig_extents)
        ax.set_ylim(_fig_extents)

        for l in range(self.num_layers):
            ax.add_artist(self.layers[l])
        _fig.savefig('detector_layers.png')

    def plot_detector_hits(self):
        """Plot the detector with hits."""
        self.plot_detector()

        plot.scatter(self.hits_x, self.hits_y, s=10)
        plot.xlabel('x')
        plot.ylabel('y')
        plot.savefig('detector_layers_hits.png')
        plot.show()

    def set_hits(self, hits_x, hits_y):
        """Sets hits on the detector layers.
        
        Sets x-y hit coordinates to the detector layers. Hits set with
        this method overwrites existing or previously set hits.

        Args:
            hits_x: List of x-coordinates of hits
            hits_y: List of y-coordinates of hits
        """
        try:
            assert len(hits_x) == len(
                hits_y
            ), 'SimpleDetector add_hits()\tERROR len(hits_x) != len(hits_y)'
            self.num_hits = len(hits_x)
            self.hits_x = hits_x
            self.hits_y = hits_y
        except AssertionError as msg:
            print(msg)


def combine_lists(list1, list2):
    """Combines two lists into a list of tuples."""
    return [(list1[i], list2[i]) for i in range(len(list1))]
