import seaborn as sns
from loguru import logger

from src.viz.viz_utils import get_font_scaler


def blank_subplot(ax_in, viz_cfg):
    """Turn off all axis elements to create a blank subplot placeholder.

    Parameters
    ----------
    ax_in : matplotlib.axes.Axes
        Axes instance to blank out.
    viz_cfg : DictConfig
        Visualization config (accepted for API consistency, unused).
    """
    ax_in.axis("off")


def style_timeseries_ax(ax_in, title_str, y_lims, legend_on=False, font_scaler=1.0):
    """Apply standard PLR time-series styling to an axes.

    Set axis labels (Time / Pupil Constriction), despine, configure legend
    visibility, apply y-limits, and format the title.

    Parameters
    ----------
    ax_in : matplotlib.axes.Axes
        Axes instance to style.
    title_str : str
        Bold left-aligned title text.
    y_lims : tuple of float or None
        ``(ymin, ymax)`` limits. Pass ``None`` to leave auto-scaled.
    legend_on : bool, optional
        Whether the legend should remain visible. Default is ``False``.
    font_scaler : float, optional
        Multiplier applied to base font sizes. Default is ``1.0``.
    """
    # https://seaborn.pydata.org/tutorial/aesthetics.html#removing-axes-spines
    try:
        sns.despine(ax=ax_in, offset=2)
    except Exception as e:
        logger.error(f"Error in using Seaborn despine: {e}")

    # Styling
    ax_in.set_xlabel("Time [s]", fontsize=int(6 * font_scaler))
    ax_in.set_ylabel("Pupil Constriction [%]", fontsize=int(6 * font_scaler))
    ax_in.legend(loc="best", fontsize=int(7 * font_scaler), framealpha=0.0)
    if not legend_on:
        ax_in.get_legend().set_visible(False)

    # https://stackoverflow.com/a/18962217/6412152 to-update with the font face name
    ax_in.set_title(
        f"{title_str}",
        y=1,
        loc="left",
        fontsize=int(9 * font_scaler),
        fontweight="bold",
    )
    if y_lims is not None:
        ax_in.set_ylim(y_lims)
    ax_in.tick_params(labelsize=int(6 * font_scaler))


def style_distribution_plot(ax_in, title_str, viz_cfg):
    """Apply standard distribution-plot styling (title and despine) to an axes.

    Parameters
    ----------
    ax_in : matplotlib.axes.Axes
        Axes instance to style.
    title_str : str
        Bold left-aligned title text.
    viz_cfg : DictConfig
        Visualization config used to derive the font scaler.
    """
    ax_in.set_title(
        f"{title_str}",
        y=1,
        loc="left",
        fontsize=int(9 * get_font_scaler(viz_cfg)),
        fontweight="bold",
    )
    try:
        sns.despine(ax=ax_in, offset=2)
    except Exception as e:
        logger.error(f"Error in using Seaborn despine: {e}")
