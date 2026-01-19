from loguru import logger
import seaborn as sns

from src.viz.viz_utils import get_font_scaler


def blank_subplot(ax_in, viz_cfg):
    ax_in.axis("off")


def style_timeseries_ax(ax_in, title_str, y_lims, legend_on=False, font_scaler=1.0):
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
