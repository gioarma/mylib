import bokeh.models as bm
import numpy as np
from holoviews.plotting.util import process_cmap
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class wlength_formatter:
    '''
    Produces a top axis in a holoviews plot containing the wavelength values (nm) corresponding to the energy values (eV) on the bottom axis

    dx: Spacing between wavelengths. Default = 100 (nm)

    formatter_auto: Creates the additional wavelength axis with automatic ticks

    formatter_dx: Creates the additional wavelength axis with ticks spaced by dx

    ##########
    How to use
    ##########

    from mylib.plotting import wlength_formatter as wf
    wlength_axis = wf(dx = 150).formatter_dx
    df.plot().opts(hooks = [wlength_axis])
    '''


    def __init__(self, dx = 100):
        self.dx = dx



    def formatter_auto(self, plot, element):
        try: en = element.data.iloc[:,0]                      # Energy values, first column of the dataset related to the plot
        except: en = element.items()[0][1].data.iloc[:,0]     # In the case of Overlays: items()[0][1] is x axis of first Curve in Overlay
        p = plot.state
        p.extra_x_ranges = {"twinx": bm.Range1d(start=en.iloc[0], end=en.iloc[-1])}                   # Create x range
        p.add_layout(bm.LinearAxis(x_range_name="twinx", axis_label = 'Wavelength (nm)'), 'above')    # Add the top axis
        p.xaxis[0].formatter = bm.FuncTickFormatter(code = """return (1239.8/tick).toFixed(0)""")     # Display wavelength values instead of energy ones
        p.xaxis[0].axis_label_text_font_size = p.xaxis[1].axis_label_text_font_size                   # Set same label font size as mirror axis
        p.xaxis[0].major_label_text_font_size = p.xaxis[1].major_label_text_font_size                 # Set same tick font size as mirror axis


    def formatter_dx(self, plot, element):
        try: en = element.data.iloc[:,0]                      # Energy values, first column of the dataset related to the plot
        except: en = element.items()[0][1].data.iloc[:,0]     # In the case of Overlays: items()[0][1] is x axis of first Curve in Overlay
        wlength = 1239.8/en                                   # Wavelength values
        ticks = [1239.8/(w*100) for w \
               in np.arange(wlength.iloc[0]//100, wlength.iloc[-1]//100, -self.dx/100)]              # Create the list of energy values where we want to display the lambda value of

        p = plot.state
        p.extra_x_ranges = {"twinx": bm.Range1d(start=en.iloc[0], end=en.iloc[-1])}                 # Create x range
        p.add_layout(bm.LinearAxis(x_range_name="twinx", axis_label = 'Wavelength (nm)'), 'above')  # Add the top axis
        p.xaxis[0].formatter = bm.FuncTickFormatter(code = """return (1239.8/tick).toFixed(0)""")   # Display wavelength values instead of energy ones
        p.xaxis[0].ticker = bm.FixedTicker(ticks=ticks)                                             # Display only desired ticks
        p.xaxis[0].axis_label_text_font_size = p.xaxis[1].axis_label_text_font_size                 # Set same label font size as mirror axis
        p.xaxis[0].major_label_text_font_size = p.xaxis[1].major_label_text_font_size               # Set same tick font size as mirror axis








class energy_formatter:

    '''
    Produces a top axis in a holoviews plot containing the energy values (nm) corresponding to the wavelength values (eV) on the bottom axis

    dx: Spacing between energies. Default = 1 (eV)

    formatter_auto: Creates the additional energy axis with automatic ticks

    formatter_dx: Creates the additional energy axis with ticks spaced by dx

    ##########
    How to use
    ##########

    from mylib.plotting import energy_formatter as wf
    energy_axis = wf(dx = 0.5).formatter_dx
    df.plot().opts(hooks = [wlength_axis])
    '''
    def __init__(self, dx = 1):
        self.dx = dx

    def formatter_auto(self, plot, element):
        try: wl = element.data.iloc[:,0]                      # Wavelength values, first column of the dataset related to the plot
        except: wl = element.items()[0][1].data.iloc[:,0]     # In the case of Overlays: items()[0][1] is x axis of first Curve in Overlay
        p = plot.state
        p.extra_x_ranges = {"twinx": bm.Range1d(start=wl.iloc[0], end=wl.iloc[-1])}                   # Create x range
        p.add_layout(bm.LinearAxis(x_range_name="twinx", axis_label = 'Energy (eV)'), 'above')        # Add the top axis
        p.xaxis[0].formatter = bm.FuncTickFormatter(code = """return (1239.8/tick).toFixed(2)""")     # Display energy values instead of wavelength ones



    def formatter_dx(self, plot, element):
        try: wl = element.data.iloc[:,0]                      # Wavelength values, first column of the dataset related to the plot
        except: wl = element.items()[0][1].data.iloc[:,0]     # In the case of Overlays: items()[0][1] is x axis of first Curve in Overlay
        en = 1239.8/wl                                        # Energy values
        ticks = [1239.8/(e) for e \
                 in np.arange(en.iloc[0]//1, en.iloc[-1]//1, -self.dx)]              # Create the list of energy values where we want to display the lambda value of
        p = plot.state
        p.extra_x_ranges = {"twinx": bm.Range1d(start=wl.iloc[0], end=wl.iloc[-1])}                 # Create x range
        p.add_layout(bm.LinearAxis(x_range_name="twinx", axis_label = 'Energy (eV)'), 'above')      # Add the top axis
        p.xaxis[0].formatter = bm.FuncTickFormatter(code = """return (1239.8/tick).toFixed(1)""")   # Display wavelength values instead of energy ones
        p.xaxis[0].ticker = bm.FixedTicker(ticks=ticks)                                             # Display only desired ticks
        p.xaxis[0].axis_label_text_font_size = p.xaxis[1].axis_label_text_font_size                 # Set same label font size as mirror axis
        p.xaxis[0].major_label_text_font_size = p.xaxis[1].major_label_text_font_size               # Set same tick font size as mirror axis


########################################################################################################

def cmap (ncols, colormap = 'fire_r'):
    '''
    Returns a colormap (list of colors) with the specified number of colors.

    ncols: Number of colors
    colormap: name of the colormap (see http://holoviews.org/user_guide/Colormaps.html)
    '''
    if '#ffffff' in process_cmap(colormap, ncols):      # Check if white is in the list, in this case I increase by one the number of colors so that I stretch the color list and then remove the white color
        colormap = [c for c in process_cmap(colormap, ncols+1) if not c=='#ffffff']
        return colormap
    else:
        return process_cmap(colormap, ncols)




def truncate_cmap(cmap_name, minval=0.0, maxval=1.0, n=100):
    '''
    Use with matplotlib.
    Truncates a colormap from minval to maxval.
    Returns a subset of the input colormap

    cmap_name: string with matplotlib colormap name
    minval: float between 0 and 1, it is the minimum value of the colormap range to be selected
    maxval: float between 0 and 1, it is the maximum value of the colormap range to be selected
    n: number of colors in which the initial colormap is divided before selecting the range
    The
    '''
    cmap=plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
