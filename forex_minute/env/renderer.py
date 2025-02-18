import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional
import pygame

class ForexMinuteRenderer:
    """Renderer for the ForexMinute environment."""
    
    def __init__(self, figure_size: tuple = (6.4, 4.8)):
        """Initialize the renderer."""
        self.figure_size = figure_size
        self.fig = None
        self.ax = None
        self.window = None
        self.clock = None
        
    def _setup_plot(self) -> None:
        """Set up the matplotlib figure and axis."""
        if self.fig is not None:
            plt.close(self.fig)
            
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis_date()
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
    def _plot_price_history(self, state: Dict[str, Any]) -> None:
        """Plot price history line."""
        self.ax.plot(
            state['timestamps'],
            state['closes'],
            label='Close Prices',
            color='blue',
            linewidth=1.5
        )
        
        # Add current price point
        self.ax.scatter(
            state['timestamps'][-1],
            state['closes'][-1],
            color='blue',
            s=100,
            zorder=5
        )
        
    def _plot_position(self, state: Dict[str, Any]) -> None:
        """Plot current position information."""
        if state['position_type'] != 1:  # If there's an open position
            position_color = 'red' if state['position_type'] == 0 else 'green'
            position_label = 'Sell Position' if state['position_type'] == 0 else 'Buy Position'
            
            # Plot position line
            self.ax.axhline(
                y=state['position_value'],
                color=position_color,
                linestyle='--',
                label=position_label,
                alpha=0.5
            )
            
            # Calculate current P&L
            current_pnl = (
                (state['current_price'] - state['position_value'])
                if state['position_type'] == 2
                else (state['position_value'] - state['current_price'])
                if state['position_type'] == 0
                else 0
            )
            
            # Add P&L text
            plt.text(
                0.02, 0.98,
                f'P&L: {current_pnl:.5f}',
                transform=self.ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='none'
                ),
                color='green' if current_pnl > 0 else 'red'
            )
            
    def render(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Render the current state.
        
        Args:
            state: Dictionary containing:
                - timestamps: List of datetime objects
                - closes: List of closing prices
                - position_type: Current position type (0=sell, 1=none, 2=buy)
                - position_value: Current position value
                - current_price: Current market price
        
        Returns:
            numpy.ndarray: RGB array of the rendered image
        """
        self._setup_plot()
        
        # Plot main elements
        self._plot_price_history(state)
        self._plot_position(state)
        
        # Customize plot
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left')
        self.ax.set_title('Forex Minute Trading')
        plt.tight_layout()
        
        # Convert plot to RGB array
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.renderer.buffer_rgba())
        
        # Clean up
        plt.close(self.fig)
        
        return img[:, :, :3]  # Return RGB (remove alpha channel)
        
    def close(self) -> None:
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()