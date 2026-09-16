import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional, Any
from CFM.core.band import Band
from CFM.core.qot_estimator import ParameterBuilder


class GSNRPlotter:
    """
    Visualization tool for Generalized Signal-to-Noise Ratio (GSNR) spectral curves.

    Generates multi-band Plotly figures illustrating linear (ASE), nonlinear (NLI),
    and total GSNR distributions across frequency grids for specific link spans.

    Attributes:
    -----------
        nuu (float): Reference frequency in Hz[cite: 5, 6].
        CCFV (np.ndarray): Channel carrier frequency offsets relative to nuu in Hz[cite: 5, 6].
        curves (List[Dict[str, Any]]): Definitions of GSNR curves to display[cite: 5].
        bands (List[Band]): Optical transmission bands defining grid segments[cite: 5].
    """

    def __init__(self, 
                 parameter_builder: ParameterBuilder, 
                 curves: List[Dict[str, Any]], 
                 bands: List[Band]):
        """
        Initialize the GSNRPlotter with grid parameters, curve values, and bands.

        Args:
        ---------
            parameter_builder (ParameterBuilder): 
                Parameter container holding reference frequency (nuu) and carrier frequency vectors (CCFV)[cite: 5].
            curves (List[Dict[str, Any]]): 
                List of curve dictionaries, where each dictionary contains:
                - "name" (str): Label for the curve legend[cite: 1, 5].
                - "values" (np.ndarray): 2D array of GSNR values with shape (N_ss, N_c) in dB[cite: 5].
                - "color" (str, optional): CSS color or hex string for the trace line[cite: 1, 5].
            bands (List[Band]): 
                List of Band objects in the identical sequence used to assemble the frequency grid[cite: 5].

        Example:
        ---------
        >>> from CFM.core.post_process import GSNRPlotter
        >>> curves = [
        ...     {"name": "GSNR_total", "values": ot, "color": "blue"},
        ...     {"name": "GSNR_linear", "values": oa, "color": "green"},
        ...     {"name": "GSNR_nonlinear", "values": on, "color": "red"}
        ... ]
        >>> plotter = GSNRPlotter(parameter_builder=params, curves=curves, bands=bands)
        >>> fig = plotter.plot(N_s_max=0, matlab_indexing=False)
        >>> fig.show()
        """
        self.nuu = parameter_builder.nuu
        self.CCFV = parameter_builder.CCFV
        self.curves = curves
        self.bands = bands

    def plot(self, N_s_max: int = 0, matlab_indexing: bool = False) -> go.Figure:
        """
        Generate an interactive Plotly scatter plot of GSNR curves across optical frequencies.

        Args:
        ---------
            N_s_max (int, optional): 
                Span index to extract and plot. Default is 0[cite: 1, 5].
            matlab_indexing (bool, optional): 
                If True, treats N_s_max as a 1-based MATLAB index and converts it 
                to 0-based indexing. Default is False[cite: 5].

        Returns:
        ---------
            go.Figure: 
                Plotly figure containing segmented transmission band traces.

        Example:
        ---------
        >>> fig = plotter.plot(N_s_max=1, matlab_indexing=True)
        >>> fig.show()
        """
        if matlab_indexing:
            N_s_max -= 1

        freq_THz = (self.nuu + self.CCFV) * 1e-12

        fig = go.Figure()

        start = 0
        band_indices = []

        for band in self.bands:
            end = start + band.num_channels
            band_indices.append(np.arange(start, end))
            start = end

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
    """
    Heatmap visualization tool for connection-level modulation format allocations.

    Plots modulation format indices across network connections and optical channel 
    indices for chosen shortest path candidates (kSP).

    Attributes:
    -----------
        modulation_connection (np.ndarray): 
            3D matrix of modulation format indices with shape (num_connections, num_channels, kSP)[cite: 5].
    """

    def __init__(self, modulation_connection: np.ndarray):
        """
        Initialize the ModulationConnectionPlotter with routing modulation profiles.

        Args:
        ---------
            modulation_connection (np.ndarray): 
                3D array of modulation format allocations structured as 
                [connection_index, channel_index, path_index][cite: 5].

        Example:
        ---------
        >>> from CFM.core.post_process import ModulationConnectionPlotter
        >>> # modulation_matrix shape: (num_connections, num_channels, kSP)
        >>> plotter = ModulationConnectionPlotter(modulation_connection=modulation_matrix)
        """
        self.modulation_connection = modulation_connection

    def plot(self, 
                k_index: int = 0, 
                title: Optional[str] = None, 
                matlab_indexing: bool = False) -> go.Figure:
        """
        Render a 2D heatmap showing modulation index distribution for a specific path index.

        Args:
        ---------
            k_index (int, optional): 
                Index of the shortest path candidate (kSP) to visualize. Default is 0[cite: 5].
            title (str, optional): 
                Custom title for the figure. If None, a default path-dependent title is used[cite: 5].
            matlab_indexing (bool, optional): 
                If True, treats k_index as a 1-based MATLAB index and converts it 
                to 0-based indexing. Default is False[cite: 5].

        Returns:
        ---------
            go.Figure: 
                Plotly heatmap figure displaying the modulation index matrix.

        Example:
        ---------
        >>> fig = plotter.plot(k_index=1, title="Path 1 Modulation Formats", matlab_indexing=True)
        >>> fig.show()
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

        fig.update_yaxes(autorange="reversed")

        fig.update_xaxes(showgrid=True, showline=True, linewidth=2,
                            linecolor="black", mirror=True)
        fig.update_yaxes(showgrid=True, showline=True, linewidth=2,
                            linecolor="black", mirror=True)

        return fig