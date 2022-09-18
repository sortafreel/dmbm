import re
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from colour import Color
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame, Series

REGION_NAMES = {
    "Київська область": "Kyiv region",
    "Вінницька область": "Vinnytsia region",
    "Житомирська область": "Zhytomyr region",
    "Кіровоградська область": "Kirovohrad region",
    "ІваноФранківська область": "Ivano-Frankivsk region",
    "Львівська область": "Lviv region",
    "Рівненська область": "Rivne region",
    "Волинська область": "Volyn region",
    "м.Київ": "Kyiv region",
    "Тернопільська область": "Ternopil region",
    "Чернігівська область": "Chernihiv region",
    "Дніпропетровська область": "Dnipropetrovsk region",
    "Харківська область": "Kharkiv region",
    "Хмельницька область": "Khmelnytskyi region",
    "Черкаська область": "Cherkasy region",
    "Одеська область": "Odesa region",
    "Запорізька область": "Zaporizhzhia region",
    "Чернівецька область": "Chernivtsi region",
    "Закарпатська область": "Transcarpathian region",
    "Сумська область": "Sumy region",
    "Полтавська область": "Poltava region",
    "Миколаївська область": "Mykolaiv region",
    "Донецька область": "Donetsk region",
    "Луганська область": "Luhansk region",
    "Херсонська область": "Kherson region",
    "Крим": "Crimea",
}

MATCH_MAP_NAMES = {
    'Vinnytsia region': 'Vinnytska',
    'Zhytomyr region': 'Zhytomyrska',
    'Kirovohrad region': 'Kirovohradska',
    'Ivano-Frankivsk region': 'Ivano-Frankivska',
    'Lviv region': 'Lvivska',
    'Rivne region': 'Rivnenska',
    'Volyn region': 'Volynska',
    'Kyiv region': 'Kyivska',
    'Ternopil region': 'Ternopilska',
    'Chernihiv region': 'Chernihivska',
    'Dnipropetrovsk region': 'Dnipropetrovska',
    'Kharkiv region': 'Kharkivska',
    'Khmelnytskyi region': 'Khmelnytska',
    'Cherkasy region': 'Cherkaska',
    'Odesa region': 'Odeska',
    'Zaporizhzhia region': 'Zaporizka',
    'Chernivtsi region': 'Chernivetska',
    'Transcarpathian region': 'Zakarpatska',
    'Sumy region': 'Sumska',
    'Poltava region': 'Poltavska',
    'Mykolaiv region': 'Mykolaivska',
    'Donetsk region': 'Donetska',
    'Luhansk region': 'Luhanska',
    'Kherson region': 'Khersonska',
    'Crimea': 'Avtonomna Respublika Krym',
    'Sevastopol region': 'Sevastopilska'
}


class AnalysisArea(Enum):
    LVIV = "Lviv"
    ALL_UKRAINE = "All Ukraine"


def size_plot(plot_len: int, plot_count: int = 1) -> Tuple[int, int]:
    """
    Calculate plot size.
    """
    plot_height = plot_len * 0.35
    # Adjust height to the minimal sensible one
    if plot_height < 5:
        plot_height = 5
    plot_width = plot_count * 11
    return plot_width, plot_height


def color_plot_adaptive(
        plot_column: Series, first_hex: str, last_hex: str
) -> List[str]:
    """
    Color the plot. Same values should have the same color.
    """
    plot_column_sorted = plot_column.sort_values()
    unique_values = list(plot_column_sorted.unique())
    plot_len_unique = len(unique_values)

    # Create colors
    first_color = Color(first_hex)
    last_color = Color(last_hex)
    plot_colors = []

    # Create palette based on the number of unique values
    for color in list(first_color.range_to(last_color, plot_len_unique)):
        plot_colors.append(str(color))

    # Manually replace first/last hex for compatibility
    plot_colors[0] = first_hex
    plot_colors[-1] = last_hex

    # Assign colors
    colors_dict = {}
    for variant in zip(plot_colors, unique_values):
        colors_dict[variant[1]] = variant[0]
    plot_colors_adaptive = []
    for value in plot_column.values:
        plot_colors_adaptive.append(colors_dict[value])

    return plot_colors_adaptive


def create_plot(
        plot_df: pd.DataFrame,
        plot_x: str,
        plot_y: str,
        first_hex: str,
        last_hex: str,
        sort_plot: bool = False,
        plot_title: Optional[str] = None,
        y_label: str = "",
        x_label: str = "",
        plot_count: int = 1,
        results_limit: bool = False,
        save: bool = False,
        animate: bool = False,
        animation_duration: float = 1.0,
        animation_fps: int = 60,
        plot_color_column: Optional[str] = None,
        plot_color_column_label: Optional[str] = None
) -> None:
    # If title is not defined - use the y-column name
    if plot_title is None:
        plot_title = f"{plot_y}_{plot_x}"

    plot_data = plot_df.reset_index()
    if sort_plot is not False:
        plot_data = plot_data.sort_values(by=plot_y)

    # Convert names (plot x) to string to be able to visualize values (plot y) consistently
    plot_data[plot_x] = plot_data[plot_x].astype(str)

    # Limit data, if required
    if results_limit is not False:
        # Reverse data, to get first N results
        plot_data = plot_data.iloc[::-1][:results_limit]
        # Reverse back for better visualization, fishy as hell, but is required
        plot_data = plot_data.iloc[::-1]

    # Define figure and values
    plot_size = size_plot(len(plot_data), plot_count)
    fig = plt.figure(figsize=plot_size)
    axes = fig.add_subplot(1, 1, 1)
    names = plot_data[plot_x].tolist()
    values = plot_data[plot_y].tolist()
    max_value = max(values)
    # Make plot a bit wider than needed
    axes.set_xlim(0, max_value * 1.05)
    axes.set_facecolor("#141414")

    # Define font and titles
    csfont = {"fontname": "Ubuntu"}
    plt.rcParams.update({"font.size": 12})
    plt.title(plot_title, fontsize=20, **csfont)
    plt.xlabel(x_label, fontsize=16, **csfont)
    plt.ylabel(y_label, fontsize=16, **csfont)

    # Process colors and the legend
    if plot_color_column is not None and plot_color_column_label is not None:
        import matplotlib.patches as mpatches
        plot_colors = color_plot_adaptive(plot_data[plot_color_column], first_hex, last_hex)
        red_patch = mpatches.Patch(color=last_hex, label=plot_color_column_label)
        axes.legend(handles=[red_patch], loc=4)
    else:
        plot_colors = color_plot_adaptive(plot_data[plot_y], first_hex, last_hex)

    # Save data
    if animate:
        if not save:
            raise Exception(
                "Animated plots need to be saved to visualize (`save` argument should be `True`)."
            )

        def animate(step_i: int):
            a_names = animation_steps[step_i][0]
            a_values = animation_steps[step_i][1]
            plt.barh(a_names, a_values, color=plot_colors)

        plot_filename = safe_filename(plot_title) + ".gif"
        animation_steps = calculate_animation_steps(
            names, values, animation_duration, animation_fps
        )
        ani = FuncAnimation(fig, animate, int(animation_duration * animation_fps))
        ani.save(
            f"images/{plot_filename}", dpi=300, writer=PillowWriter(fps=animation_fps)
        )
        return

    # Display plot, placed here, so it won't affect animation
    plt.barh(names, values, color=plot_colors)
    if not save:
        return
    plot_filename = safe_filename(plot_title) + ".png"
    plt.savefig(f"images/{plot_filename}", bbox_inches="tight")
    return


def calculate_animation_steps(
        ar_names: List[Any],
        ar_values: List[int],
        animation_duration: float,
        animation_fps: int,
) -> List[List[Union[Any, int]]]:
    steps = []

    for i in range(int(animation_duration * animation_fps)):
        if i == 0:
            step_values = [0 for _ in range(len(ar_values))]
            steps.append([ar_names, step_values])
            continue
        step_values = [
            x + ar_values[i] / (animation_duration * animation_fps)
            for i, x in enumerate(steps[-1][1])
        ]
        steps.append((ar_names, step_values))
    return steps


def stringify_messages(raw_message: List[Union[str, Dict]]):
    message = []
    for part in raw_message:
        if isinstance(part, dict):
            message.append(part.get("text", ""))
        else:
            message.append(part)
    return "".join(message)


def timedelta_to_hours(delta: timedelta) -> float:
    hours = 0
    hours += delta.days * 24
    hours += delta.seconds / (60 * 60)
    return hours


def group_alerts_by_hour(all_alarms_df: DataFrame) -> DataFrame:
    agg_params = {"start_datetime": ["count"], "duration_hours": ["sum"]}
    per_hour_df = all_alarms_df.groupby(all_alarms_df["start_datetime"].dt.hour).agg(
        agg_params
    )
    per_hour_df.reset_index(inplace=True)
    per_hour_df.columns = ["hour", "alerts_per_hour", "duration_hours"]
    per_hour_df["hour"] = per_hour_df["hour"].apply(lambda x: "{:02d}:00".format(x))
    return per_hour_df


def safe_filename(init_filename: str) -> str:
    return "".join(
        [c for c in init_filename if c.isalpha() or c.isdigit() or c == " "]
    ).rstrip()


def generate_sleep_periods() -> Dict[int, Dict[str, Union[List, int]]]:
    sleep_periods = {}
    for i in range(24):
        period = [x if x < 24 else x - 24 for x in range(i, i + 9)]
        if i == 0:
            sleep_periods[24] = {"affected_hours": period, "alerts_per_sleep_period": 0, "hours_per_sleep_period": 0}
            continue
        sleep_periods[i] = {"affected_hours": period, "alerts_per_sleep_period": 0, "hours_per_sleep_period": 0}
    return sleep_periods


def combine_hours_into_sleep_periods(per_hour_df: DataFrame) -> DataFrame:
    # TODO The code could be way better
    sleep_periods = generate_sleep_periods()
    # Collect amount of alerts per hour
    for key, value in sleep_periods.items():
        for hour in value["affected_hours"]:
            if hour == 0:
                hour == 24
            related_rows = per_hour_df[per_hour_df["hour"] == "{:02d}:00".format(hour)]
            if len(related_rows) == 0:
                continue
            sleep_periods[key]["alerts_per_sleep_period"] += related_rows.iloc[0][
                "alerts_per_hour"
            ]
            sleep_periods[key]["hours_per_sleep_period"] += related_rows.iloc[0][
                "duration_hours"
            ]
    per_sleep_period_df = pd.DataFrame(sleep_periods).transpose()
    per_sleep_period_df["sleep_period"] = per_sleep_period_df["affected_hours"].apply(
        lambda x: "{:02d}:00-{:02d}:00".format(x[0], x[-1])
    )
    # Calculate percentage from total alerts
    total_alerts = per_hour_df["alerts_per_hour"].sum()
    per_sleep_period_df["alerts_percetage_per_sleep_period"] = per_sleep_period_df[
        "alerts_per_sleep_period"
    ].apply(lambda x: x / total_alerts * 100)
    return per_sleep_period_df


def _marker_match(markers: List[str], message: str) -> bool:
    for marker in markers:
        if re.match(marker, message):
            return True
    else:
        return False


def define_message_type(start_markers: List[str], end_markers: List[str]) -> Callable:
    def inner(message: str) -> bool:
        if not message:
            return "info"
        if _marker_match(start_markers, message):
            return "start"
        elif _marker_match(end_markers, message):
            return "end"
        else:
            return "info"

    return inner


def define_region(message: str) -> Optional[str]:
    search_group = re.findall(r"\n#(.*)$", message)
    if len(search_group) == 0:
        print(f"Can't identify region from the message: {message}")
        return
    region = search_group[0]
    if not region:
        print(f"Can't identify region from the message: {message}")
        return
    # Make region more reabable
    region = re.sub(r"^м_", "м.", region).replace("_", " ")
    # Translate
    return REGION_NAMES.get(region, region)


def generate_cmap(steps: int, first_hex: str, last_hex: str) -> LinearSegmentedColormap:
    palette = generate_color_palette(steps, first_hex, last_hex)
    # Generate cmap
    cmap = LinearSegmentedColormap.from_list("ukraine", [x.rgb for x in palette])
    return cmap


def generate_color_palette(steps: int, first_hex: str, last_hex: str) -> list[Color]:
    # Create colors
    first_color = Color(first_hex)
    last_color = Color(last_hex)
    plot_colors = []
    # Create palette based on the number of unique values
    for color in list(first_color.range_to(last_color, steps)):
        plot_colors.append(color)
    return plot_colors
