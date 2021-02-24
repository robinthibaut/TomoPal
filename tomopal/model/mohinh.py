#  Copyright (c) 2020. Robin Thibaut, Ghent University

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.ticker import LogFormatter
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import TextBox


def find_norm(l):  # Convert values to linear space to facilitate visualization!
    uv = list(dict.fromkeys(l))
    uv.sort()

    ls = np.linspace(0, 1, len(uv))

    nl = []

    for i in range(len(l)):
        c = 0
        for j in range(len(uv)):
            if l[i] <= uv[j] and not c:
                nl.append(ls[j])
                c += 1
    return nl


class ModelMaker(object):

    def __init__(self,
                 model_name=None,
                 centerxy=np.array([]),
                 blocks=np.array([]),
                 values=np.array([]),
                 values_log=0,
                 bck=1):

        """
        :param model_name: path to the file to be created containing the output
        :param centerxy: x-y coordinates of the center of the cells
        :param blocks: coordinates of the corners of the different blocks
        :param values: array containing the value assigned to each block
        :param values_log: flag indicating if values should be log transformed or not
        :param bck: background value
        """

        if centerxy.any():  # If block center coordinates are provided, plot them.
            xs = centerxy[:, 0]
            ys = centerxy[:, 1]
            fig, ax = plt.subplots()
            pts = ax.scatter(xs, ys, c="black", alpha=0.5)
        else:  # Else, if only blocks are provided, the centers are computed based on the mean of the coordinates of
            # their corners
            try:
                centerxy = np.array([np.mean(b, axis=0) for b in blocks])  # Mean computed
                xs = centerxy[:, 0]
                ys = centerxy[:, 1]
                fig, ax = plt.subplots()
                pts = ax.scatter(xs, ys, c="black", alpha=0.5)
            except Exception as e:
                print(e)
                exit()

        pts.set_zorder(2)  # I use this option to put the points in first plan, as to always see them.

        self.points = centerxy

        self.bck = bck

        self.model_name = model_name

        self.ax = ax  # Axes object

        self.blocks = blocks

        self.ax.set_title('Now setting value: 0.0')  # Initial title

        self.ax.set_facecolor((0.86, 0.86, 0.86))  # Set gray background
        plt.subplots_adjust(bottom=0.2, right=0.8)  # Adjusts plt dimensions to insert colorbar, textbox...

        self.vals = []  # List that will contain the values assigned to the different polygons

        self.cmap = cm.get_cmap('jet')  # Color map function to be later used

        axbox = plt.axes([0.4, 0.07, 0.1, 0.07])  # It is necessary to define a different axes for the box (normalized)
        # 4 - tuple of floats
        # rect = [left, bottom, width, height]. A new axes is added
        # with dimensions rect in normalized (0, 1) units using ~.Figure.add_axes on the current figure.

        self.axcb = plt.axes([0.82, 0.07, 0.015, 0.83])  # Location of colorbar for the user-defined polygons
        bounds = [0] + [1]  # Adding one bound more to have enough intervals
        ticks = [0]  # Ticks - default value
        cols = colors.ListedColormap([self.cmap(v) for v in ticks])
        cbnorm = colors.BoundaryNorm(bounds, cols.N)
        mcb = colorbar.ColorbarBase(self.axcb,
                                    cmap=cols,
                                    norm=cbnorm,
                                    boundaries=bounds,
                                    ticks=ticks,
                                    ticklocation='right',
                                    orientation='vertical')
        #
        #         textstr = """Select points in the figure by enclosing them within a polygon.
        # Press the 'esc' key to start a new polygon.
        # Try holding the 'shift' key to move all of the vertices.
        # Try holding the 'ctrl' key to move a single vertex."""
        #
        #         #axtxt = plt.axes([0.15, 0.0, 0.2, 0.15])
        #         props = dict(boxstyle='round', facecolor='green', alpha=0.5)
        #         ax.text(0, -0.2, textstr, transform=ax.transAxes, fontsize=10, bbox=props)

        self.vinput = TextBox(axbox, label=None, initial='0')  # Text box to input the value to input
        self.vinput.on_submit(self.button_submit)  # What happens when pressing enter

        self.index = []  # List that will contain the final results !

        self.canvas = self.ax.figure.canvas  # Creates a canvas from the ax of the scatter plot

        self.collection = pts  # 'collection' is the scatter plot
        self.xys = pts.get_offsets()  # Necessary for later to define if points enclosed by polygon - basically
        # equals to the - x-y coordinates of the different points.
        self.Npts = len(self.xys)  # Number of points

        # Ensure that we have separate colors for each object
        self.fc = pts.get_facecolors()  # Gets the rgb of the points

        if not values.any():
            facecolors = 'gray'
            alpha = 0.35  # Opacity of the polygons - if no values assigned - soft gray
        else:
            cmap2 = cm.get_cmap('coolwarm')
            if values_log:
                itv = 10 ** np.linspace(np.log10(min(values)), np.log10(max(values)), 12)  # Making a nice linear space
                # out ouf log values to represent some ticks on the colorbar
                norm2 = colors.LogNorm(vmin=min(values), vmax=max(values))  # Log norm for color bar
                formatter = LogFormatter(10, labelOnlyBase=False)  # Necessary arg to produce the color scale
            else:
                itv = np.linspace(min(values), max(values), 8)  # Linear space
                norm2 = colors.Normalize(vmin=min(values), vmax=max(values))
                formatter = None
            facecolors = [cmap2(norm2(v)) for v in values]  # Individual color of each polygon
            alpha = 0.6  # Opacity of the polygons

            # Colorbar if initial values present - plotting a nice color bar
            ticks2 = [round(v, 1) for v in itv]
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
            axcb1 = plt.axes([0.15, 0.07, 0.015, 0.83])
            cb1 = colorbar.ColorbarBase(axcb1,
                                        cmap=cmap2,
                                        norm=norm2,
                                        ticks=ticks2,
                                        boundaries=None,
                                        ticklocation='left',
                                        format=formatter,
                                        orientation='vertical')
            cb1.set_ticklabels(ticks2)  # Setting the proper labels

        if self.blocks.any():  # If blocks are provided. I should change this as the direction this code is going is
            # to provide blocks by default

            xs = self.blocks[:, :, 0]  # x-coordinates blocks corners
            ys = self.blocks[:, :, 1]  # y-coordinates blocks corners

            patches = []  # Coloring the blocks, in gray or with different colors
            for b in blocks:
                polygon = Polygon(b, closed=True)
                patches.append(polygon)
            p = PatchCollection(patches, alpha=alpha, facecolors=facecolors, edgecolors='black')
            p.set_zorder(0)
            self.ax.add_collection(p)

            padx = (xs.max() - xs.min()) * 0.05  # 5% padding in x-direction for visualization
            pady = (ys.max() - ys.min()) * 0.05  # 5% padding
            self.ax.set_xlim(xs.min() - padx, xs.max() + padx)
            self.ax.set_ylim(ys.min() - pady, ys.max() + pady)

        self.collection.set_facecolors(facecolors)  # Coloring the points

        self.poly = PolygonSelector(self.ax, self.onselect)  # Polygon selector object
        self.ind = []  # Initiates the ind list for a new polygon!

        def handle_close(evt):  # Function that disconnects the PolygonSelector
            self.disconnect()

        self.canvas.mpl_connect('close_event', handle_close)  # When closing window, finishes the job

        self.final_results = np.ones(len(self.points)) * self.bck  # Final results, array filled with background value.

    def button_submit(self, text):
        """
        Controls what happens when pressing Enter in the text box
        :param text: Entered value, supposed to be a number
        :return:
        """
        try:
            tt = text.translate({ord(c): "." for c in "!@#$%^&*()[]{};:,/<>?\\|`~=_+"})  # In case the user inputs a
            # float number with a character different than '.', this will convert it.
            z = float(tt)  # When pressing enter in the text box, ads the value to 'vals'
        except:
            z = 0
            print('Enter a number in the text box')

        self.vals.append(z)  # When pressing enter in the text box, the value will be added to the 'vals' list
        # self.vinput.set_val(z)
        self.ax.set_title('Now setting value: {}'.format(self.vals[-1]))  # Updates the title

    def onselect(self, verts):
        """
        Defines what happens when a polygon is completed

        :param verts: polygon vertices automatically returned by the Polygon Selector
        :return:
        """

        if len(self.vals) == 0:  # If the user forgets to press enter to initiate value
            self.vals.append(0)
            print('Press enter in the text box')

        path = Path(verts)  # When a polygon is completed or modified after completion, the `onselect`
        # function is called and passed a list of the vertices as ``(xdata, ydata)`` tuples.

        self.ind = np.nonzero(path.contains_points(self.xys))[0]  #

        # xycenter = self.xys[self.ind]
        # xycenter.data
        self.index.append([self.vals[-1], self.ind, verts])
        # OP : Value, Index #, XY Coordinates selected points, Polygon vtx

        current_vals = [self.index[i][0] for i in range(len(self.index))]  # Lists all values assigned to each polygon,
        # stored in the 'index' list
        vals_cb = list(dict.fromkeys(current_vals))  # Removes duplicates

        lsps = find_norm(current_vals)  # Get colors

        cc = [self.cmap(v) for v in lsps]  # Colors used for the points/blocks

        ccb = lsps.copy()  # Copy list
        ccb = list(dict.fromkeys(ccb))  # Removes duplicates
        ccb.sort()
        # List of colors for the colorbar
        cols = colors.ListedColormap([self.cmap(v) for v in ccb])

        if len(vals_cb) > 1:  # Creating a colormap fn to automatically rescale the colors of selected zones

            try:
                vals_cb.sort()
                bounds = vals_cb + [max(vals_cb) * 1.1]  # Adding one bound more to have enough intervals
                ticks = vals_cb  # Ticks labels
                cbnorm = colors.BoundaryNorm(bounds, cols.N)
                mcb = colorbar.ColorbarBase(self.axcb,
                                            cmap=cols,
                                            norm=cbnorm,
                                            boundaries=bounds,
                                            ticks=ticks,
                                            ticklocation='right',
                                            orientation='vertical')
            except:
                pass

        else:
            pass

        if not self.blocks.any():  # If no blocks, simply coloring the points accordingly
            for i in range(len(self.index)):
                self.fc[self.index[i][1]] = self.cmap(lsps[i])
            self.collection.set_facecolors(self.fc)  # updates points color

        else:
            if len(self.index) > 0:
                for i in range(len(self.index)):  # If blocks
                    bid = self.blocks[self.index[i][1]]  # Concerned mesh blocks
                    patches = []
                    for b in bid:
                        polygon = Polygon(b, closed=True)
                        patches.append(polygon)
                    p = PatchCollection(patches, alpha=1, facecolors=cc[i])
                    p.set_zorder(1)
                    self.ax.add_collection(p)
            else:
                pass

        # self.collection.set_facecolors(self.fc)  # updates points color

        self.canvas.draw()

        idx = np.copy(self.index)  # Necessary to copy to not mess with the original array

        # What you see is what you get

        for i in range(len(idx) - 1, 0, -1):  # Removes duplicates from older selections, scanning for the latest
            # polygons first to the last.
            indc = idx[i][1]
            indp = idx[i - 1][1]
            upd = np.array(list(set(indp) - set(indc)))
            self.index[i - 1][1] = upd  # Updates the oldest polygons values, removing duplicates index values

        # final_results = np.ones(len(self.points))*self.bck  # Multiplying by the background value

        for item in self.index:
            for i in item[1]:
                self.final_results[i] = item[0]

        # Now save the results
        if self.model_name:
            print('Saving file...')
            with open(self.model_name, 'w') as mm:  # The final output is a 1-column file with the values assigned,
                # in the same order as the blocks provided. The file is opened and saved every time a modification is
                # made.
                mm.write(str(int(self.Npts)) + '\n')
                [mm.write(str(fr) + '\n') for fr in self.final_results]
                mm.close()

    def disconnect(self):  # When finished
        """
        Disconnect the Polygon selector
        :return:
        """
        print('Job done')
        self.poly.disconnect_events()
