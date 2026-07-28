import numpy as np
import plotly.graph_objects as go


class GSNRPlotter:

    def __init__(self, parameter_builder, curves, bands):
        """
        Parameters
        ----------
        parameter_builder : object
            Object containing nuu and CCFV.

        curves : list of dict
            Curve definitions.

        bands : list[Band]
            List of Band objects in the same order used to build the grid.
        """

        self.nuu = parameter_builder.nuu
        self.CCFV = parameter_builder.CCFV
        self.curves = curves
        self.bands = bands

    def plot(self, N_s_max, matlab_indexing=False):

        if matlab_indexing:
            N_s_max -= 1

        freq_THz = (self.nuu + self.CCFV) * 1e-12

        fig = go.Figure()

        # compute index ranges for each band
        start = 0
        band_indices = []

        for band in self.bands:
            end = start + band.num_channels
            band_indices.append(np.arange(start, end))
            start = end

        # plot each band
        for i, ch_idx in enumerate(band_indices):

            show_legend = (i == 0)

            for curve in self.curves:

                fig.add_trace(go.Scatter(
                    x=freq_THz[ch_idx],
                    y=curve["values"][N_s_max, ch_idx],
                    mode="lines",
                    name=curve["name"],
                    line=dict(color=curve.get("color", None)),
                    showlegend=show_legend
                ))

        fig.update_layout(
            xaxis_title="optical frequency [THz]",
            yaxis_title="GSNR [dB]",
            template="plotly_white",
            legend=dict(
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom"
            )
        )

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        return fig



class ModulationConnectionPlotter:

    def __init__(self, modulation_connection):
        """
        Parameters
        ----------
        modulation_connection : np.ndarray
            3D array with shape:
            (num_connections, num_channels, kSP)

            Same structure as:
            Modulation_connection[connection, channel, path]
        """
        self.modulation_connection = modulation_connection

    def plot(self, k_index=0, title=None, matlab_indexing=False):
        """
        Plot the modulation index heatmap using plotly.

        Parameters
        ----------
        k_index : int
            Path index (MATLAB uses 1-based indexing).

        title : str
            Optional plot title.

        matlab_indexing : bool
            If True, user provides MATLAB-style indices → subtract 1.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        if matlab_indexing:
            k_index -= 1

        data = self.modulation_connection[:, :, k_index]

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                colorscale="Viridis",
                zmin=0,
                zmax=6,
                colorbar=dict(title="Modulation Index")
            )
        )

        fig.update_layout(
            title=title or f"Modulation_connection: k = {k_index + 1}",
            xaxis_title="Channel Index",
            yaxis_title="Connection Index",
            template="plotly_white",
            font=dict(family="Times New Roman", size=18),
        )

        # Match MATLAB imagesc orientation
        fig.update_yaxes(autorange="reversed")

        # Add axis borders and grid
        fig.update_xaxes(showgrid=True, showline=True, linewidth=2,
                         linecolor="black", mirror=True)
        fig.update_yaxes(showgrid=True, showline=True, linewidth=2,
                         linecolor="black", mirror=True)

        return fig