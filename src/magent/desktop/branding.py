"""The eight MAgent desktop logo concepts and persistent panel appearance.

Every concept has an icon for the always-on-top control and a horizontal lockup
for documentation or launchers.  SVGs are generated deterministically from this
module, while :func:`draw_logo` renders the same visual language on a Tk canvas
without requiring Pillow or an SVG renderer.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.files import atomic_write_text


@dataclass(slots=True, frozen=True)
class LogoTheme:
    """One selectable MAgent identity concept."""

    id: str
    name: str
    tagline: str
    description: str
    accent: str
    accent2: str
    shape: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe mapping."""
        return asdict(self)


LOGO_THEMES: tuple[LogoTheme, ...] = (
    LogoTheme(
        "orbit",
        "Orbit M",
        "Autonomous desktop",
        "Autonomy and motion",
        "#50e6ff",
        "#9b72ff",
        "orbit",
    ),
    LogoTheme(
        "core",
        "Agent Core",
        "Control core",
        "Stable, premium hexagonal mark",
        "#54eadf",
        "#8177ff",
        "core",
    ),
    LogoTheme(
        "command",
        "Command Wing",
        "Command your desktop",
        "Fast cursor-driven monogram",
        "#52e9ff",
        "#6f78ff",
        "command",
    ),
    LogoTheme(
        "neural",
        "Neural M",
        "Local intelligence",
        "Connected technical identity",
        "#53e7ff",
        "#ac70ff",
        "neural",
    ),
    LogoTheme(
        "flow",
        "Infinite Flow",
        "Always working",
        "Continuous and approachable",
        "#43efd2",
        "#a76fff",
        "flow",
    ),
    LogoTheme(
        "window",
        "Window Pilot",
        "Desktop pilot",
        "Direct desktop automation",
        "#4ceaff",
        "#8b72ff",
        "window",
    ),
    LogoTheme(
        "pulse",
        "Pulse M",
        "Ready when you are",
        "Responsive and alive",
        "#45efd1",
        "#9d70ff",
        "pulse",
    ),
    LogoTheme(
        "beacon",
        "Agent Beacon",
        "Always within reach",
        "Present and dependable",
        "#4de6ff",
        "#9972ff",
        "beacon",
    ),
)
"""Eight icon/lockup pairs offered in the desktop panel."""

DEFAULT_LOGO_THEME = "orbit"
"""Theme used before the operator makes a selection."""


def logo_theme(theme_id: str = "") -> LogoTheme:
    """Resolve a theme id, falling back to :data:`DEFAULT_LOGO_THEME`."""
    wanted = (theme_id or DEFAULT_LOGO_THEME).strip().lower()
    return next((theme for theme in LOGO_THEMES if theme.id == wanted), LOGO_THEMES[0])


def logo_ids() -> list[str]:
    """Return selectable theme ids in display order."""
    return [theme.id for theme in LOGO_THEMES]


def icon_svg(theme_id: str = "", *, size: int = 256) -> str:
    """Render a transparent-background icon SVG."""
    theme = logo_theme(theme_id)
    dimension = max(32, min(int(size), 2048))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{dimension}" height="{dimension}" '
        f'viewBox="0 0 120 120" role="img" aria-label="MAgent {html.escape(theme.name)} logo">'
        f"{_defs(theme)}{_icon_body(theme)}</svg>\n"
    )


def lockup_svg(theme_id: str = "", *, width: int = 640) -> str:
    """Render a horizontal icon and MAgent wordmark SVG."""
    theme = logo_theme(theme_id)
    output_width = max(240, min(int(width), 2048))
    height = round(output_width * 0.28)
    safe_name = html.escape(theme.name)
    safe_tagline = html.escape(theme.tagline.upper())
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{output_width}" height="{height}" '
        f'viewBox="0 0 500 140" role="img" aria-label="MAgent {safe_name} lockup">'
        f"{_defs(theme)}"
        '<g transform="translate(8 10) scale(1)">'
        f"{_icon_body(theme)}"
        "</g>"
        '<text x="150" y="77" fill="#f4f8ff" font-family="Inter,Segoe UI,sans-serif" '
        'font-size="48" font-weight="760" letter-spacing="-2">'
        f'<tspan fill="{theme.accent}">M</tspan><tspan>Agent</tspan></text>'
        '<text x="153" y="104" fill="#8fa1ba" font-family="Inter,Segoe UI,sans-serif" '
        f'font-size="11" font-weight="600" letter-spacing="3">{safe_tagline}</text>'
        "</svg>\n"
    )


def _defs(theme: LogoTheme) -> str:
    key = theme.id
    return (
        "<defs>"
        f'<linearGradient id="g-{key}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop stop-color="{theme.accent}"/><stop offset="1" stop-color="{theme.accent2}"/>'
        "</linearGradient>"
        f'<filter id="glow-{key}" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur stdDeviation="2.2" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter></defs>"
    )


def _icon_body(theme: LogoTheme) -> str:
    gradient = f"url(#g-{theme.id})"
    glow = f"url(#glow-{theme.id})"
    shell = '<circle cx="60" cy="60" r="48" fill="#091321" stroke="#233956" stroke-width="2"/>'
    m_path = (
        f'<path d="M29 84V38l31 32 31-32v46" fill="none" stroke="{gradient}" '
        f'stroke-width="9" stroke-linecap="round" stroke-linejoin="round" filter="{glow}"/>'
    )
    if theme.shape == "orbit":
        return (
            shell + f'<ellipse cx="60" cy="60" rx="53" ry="25" fill="none" stroke="{gradient}" '
            'stroke-width="3" transform="rotate(-23 60 60)" opacity=".85"/>'
            + m_path
            + f'<circle cx="106" cy="39" r="5" fill="{theme.accent}" filter="{glow}"/>'
        )
    if theme.shape == "core":
        return (
            f'<path d="M60 7 105 33v54l-45 26-45-26V33Z" fill="#091321" stroke="{gradient}" stroke-width="4"/>'
            + m_path
            + f'<circle cx="60" cy="70" r="5" fill="{theme.accent}"/>'
        )
    if theme.shape == "command":
        return (
            '<rect x="12" y="12" width="96" height="96" rx="29" fill="#091321" stroke="#263d5d" stroke-width="2"/>'
            + f'<path d="M27 83V38l33 32 22-23" fill="none" stroke="{gradient}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>'
            + f'<path d="m73 29 34 14-15 7-7 16Z" fill="{theme.accent}" stroke="#e9fdff" stroke-width="2"/>'
        )
    if theme.shape == "neural":
        nodes = "".join(
            f'<circle cx="{x}" cy="{y}" r="6" fill="#0c192a" stroke="{theme.accent}" stroke-width="3"/>'
            for x, y in ((27, 35), (27, 85), (60, 43), (60, 69), (93, 35), (93, 85))
        )
        return (
            shell
            + f'<path d="M27 85V35l33 34 33-34v50M27 35l33 8 33-8M60 43v26" fill="none" stroke="{gradient}" stroke-width="3" opacity=".75"/>'
            + nodes
        )
    if theme.shape == "flow":
        return (
            shell
            + f'<path d="M16 60c15-27 31-27 44 0s29 27 44 0c-15-27-31-27-44 0S31 87 16 60Z" fill="none" stroke="{gradient}" stroke-width="8" stroke-linecap="round"/>'
            + '<path d="M34 77V45l26 27 26-27v32" fill="none" stroke="#e5fbff" stroke-width="3" stroke-linecap="round" opacity=".8"/>'
        )
    if theme.shape == "window":
        return (
            f'<rect x="13" y="17" width="94" height="86" rx="18" fill="#091321" stroke="{gradient}" stroke-width="4"/>'
            '<path d="M13 39h94" stroke="#304a6a" stroke-width="3"/>'
            '<circle cx="27" cy="28" r="3" fill="#607895"/><circle cx="38" cy="28" r="3" fill="#607895"/><circle cx="49" cy="28" r="3" fill="#607895"/>'
            + m_path
            + f'<path d="m75 64 27 11-12 5-5 13Z" fill="#effdff" stroke="{theme.accent}" stroke-width="2"/>'
        )
    if theme.shape == "pulse":
        return (
            shell
            + m_path.replace('opacity=".8"', "")
            + f'<path d="M13 64h16l8-24 16 44 14-55 14 35h26" fill="none" stroke="{gradient}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" filter="{glow}"/>'
        )
    return (
        shell
        + f'<path d="M24 45a42 42 0 0 1 72 0M33 52a32 32 0 0 1 54 0M43 58a20 20 0 0 1 34 0" fill="none" stroke="{theme.accent}" stroke-width="4" stroke-linecap="round" opacity=".7"/>'
        + f'<circle cx="60" cy="62" r="7" fill="{gradient}"/>'
        + m_path
    )


def write_logo_assets(directory: str | Path) -> list[Path]:
    """Generate all eight icon/lockup pairs and an HTML selection gallery."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cards: list[str] = []
    for index, theme in enumerate(LOGO_THEMES, start=1):
        icon_path = root / f"{index:02d}-{theme.id}-icon.svg"
        lockup_path = root / f"{index:02d}-{theme.id}-lockup.svg"
        atomic_write_text(icon_path, icon_svg(theme.id))
        atomic_write_text(lockup_path, lockup_svg(theme.id))
        written.extend((icon_path, lockup_path))
        cards.append(
            '<article class="card">'
            f"<header><b>{index}</b><span><strong>{html.escape(theme.name)}</strong>"
            f"<small>{html.escape(theme.description)}</small></span></header>"
            '<div class="pair">'
            f'<div><img src="{icon_path.name}" alt="{html.escape(theme.name)} icon"></div>'
            f'<div><img src="{lockup_path.name}" alt="{html.escape(theme.name)} lockup"></div>'
            "</div></article>"
        )
    gallery = root / "index.html"
    atomic_write_text(gallery, _gallery_html("".join(cards)))
    written.append(gallery)
    return written


def _gallery_html(cards: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAgent logo concepts</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;padding:36px;background:radial-gradient(circle at 50% -20%,#173050,#070b14 45%);color:#f4f8ff;font:14px/1.5 Inter,system-ui,sans-serif}}main,body>header{{max-width:1180px;margin:auto}}h1{{margin:0;font-size:30px}}p{{color:#92a2b9;margin:4px 0 24px}}main{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:linear-gradient(145deg,#121d30,#0a101c);border:1px solid #243550;border-radius:20px;padding:18px}}.card header{{display:flex;gap:11px;align-items:center;margin-bottom:13px}}.card header b{{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:#1c2b44;color:#71e5ff}}strong,small{{display:block}}small{{color:#8fa0b8}}.pair{{display:grid;grid-template-columns:140px 1fr;gap:12px}}.pair div{{display:grid;place-items:center;min-height:145px;border:1px solid #20324d;border-radius:15px;background:#09111e;padding:13px}}img{{max-width:100%;max-height:125px}}@media(max-width:820px){{main{{grid-template-columns:1fr}}}}@media(max-width:480px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>MAgent logo concepts</h1><p>Eight selectable pairs: floating desktop icon and product lockup. All are bundled with MAgent.</p></header><main>{cards}</main></body></html>"""


@dataclass(slots=True)
class DesktopAppearance:
    """Small persistent state for the floating control."""

    logo_theme: str = DEFAULT_LOGO_THEME
    bubble_x: int = -1
    bubble_y: int = 90
    panel_geometry: str = "500x720"
    always_on_top: bool = True

    def __post_init__(self) -> None:
        self.logo_theme = logo_theme(self.logo_theme).id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)


class DesktopAppearanceStore:
    """Atomic JSON persistence for logo selection and window placement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> DesktopAppearance:
        """Load state, falling back safely when the file is absent or corrupt."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return DesktopAppearance()
        if not isinstance(payload, dict):
            return DesktopAppearance()
        known = {"logo_theme", "bubble_x", "bubble_y", "panel_geometry", "always_on_top"}
        try:
            return DesktopAppearance(
                **{key: value for key, value in payload.items() if key in known}
            )
        except (TypeError, ValueError):
            return DesktopAppearance()

    def save(self, state: DesktopAppearance) -> Path:
        """Persist state atomically."""
        atomic_write_text(
            self.path,
            json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        return self.path


def draw_logo(canvas: Any, theme_id: str, size: int, *, show_background: bool = True) -> None:
    """Draw one theme on a Tk-compatible canvas."""
    theme = logo_theme(theme_id)
    canvas.delete("all")
    scale = max(0.2, size / 120.0)

    def xy(*values: float) -> tuple[float, ...]:
        return tuple(value * scale for value in values)

    def line(points: tuple[float, ...], **kwargs: Any) -> Any:
        return canvas.create_line(*xy(*points), **kwargs)

    if show_background:
        canvas.create_oval(
            *xy(8, 8, 112, 112), fill="#091321", outline="#29415f", width=max(1, round(2 * scale))
        )
    width = max(3, round(8 * scale))
    if theme.shape == "core":
        canvas.create_polygon(
            *xy(60, 7, 105, 33, 105, 87, 60, 113, 15, 87, 15, 33),
            fill="#091321",
            outline=theme.accent,
            width=max(2, round(3 * scale)),
        )
    elif theme.shape == "command":
        canvas.create_polygon(
            *xy(73, 28, 108, 43, 92, 51, 85, 67),
            fill=theme.accent,
            outline="#e9fdff",
            width=max(1, round(2 * scale)),
        )
    elif theme.shape == "neural":
        line(
            (27, 85, 27, 35, 60, 69, 93, 35, 93, 85),
            fill=theme.accent2,
            width=max(2, round(3 * scale)),
        )
        line((27, 35, 60, 43, 93, 35), fill=theme.accent, width=max(1, round(2 * scale)))
        for x, y in ((27, 35), (27, 85), (60, 43), (60, 69), (93, 35), (93, 85)):
            radius = 5 * scale
            canvas.create_oval(
                x * scale - radius,
                y * scale - radius,
                x * scale + radius,
                y * scale + radius,
                fill="#0c192a",
                outline=theme.accent,
                width=max(1, round(2 * scale)),
            )
        return
    elif theme.shape == "flow":
        line(
            (16, 60, 34, 34, 60, 60, 86, 86, 104, 60, 86, 34, 60, 60, 34, 86, 16, 60),
            fill=theme.accent,
            width=width,
            smooth=True,
            capstyle="round",
        )
    elif theme.shape == "window":
        canvas.create_rectangle(
            *xy(13, 17, 107, 103),
            fill="#091321",
            outline=theme.accent,
            width=max(2, round(3 * scale)),
        )
        line((13, 39, 107, 39), fill="#304a6a", width=max(1, round(2 * scale)))
        canvas.create_polygon(
            *xy(75, 64, 102, 75, 90, 80, 85, 93), fill="#effdff", outline=theme.accent
        )
    elif theme.shape == "pulse":
        line(
            (13, 64, 29, 64, 37, 40, 53, 84, 67, 29, 81, 64, 107, 64),
            fill=theme.accent,
            width=max(2, round(5 * scale)),
            capstyle="round",
            joinstyle="round",
        )
    elif theme.shape == "beacon":
        for inset in (24, 34, 44):
            canvas.create_arc(
                *xy(inset, inset - 2, 120 - inset, 82 - inset / 2),
                start=25,
                extent=130,
                style="arc",
                outline=theme.accent,
                width=max(1, round(3 * scale)),
            )
        radius = 6 * scale
        canvas.create_oval(
            60 * scale - radius,
            61 * scale - radius,
            60 * scale + radius,
            61 * scale + radius,
            fill=theme.accent,
            outline="",
        )
    elif theme.shape == "orbit":
        canvas.create_oval(
            *xy(6, 34, 114, 86), outline=theme.accent, width=max(1, round(3 * scale))
        )
        radius = 5 * scale
        canvas.create_oval(
            104 * scale - radius,
            39 * scale - radius,
            104 * scale + radius,
            39 * scale + radius,
            fill=theme.accent,
            outline="",
        )
    line(
        (29, 84, 29, 38, 60, 70, 91, 38, 91, 84),
        fill=theme.accent2,
        width=width,
        capstyle="round",
        joinstyle="round",
    )


__all__ = [
    "DEFAULT_LOGO_THEME",
    "LOGO_THEMES",
    "DesktopAppearance",
    "DesktopAppearanceStore",
    "LogoTheme",
    "draw_logo",
    "icon_svg",
    "lockup_svg",
    "logo_ids",
    "logo_theme",
    "write_logo_assets",
]
