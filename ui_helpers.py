"""HTML/UI hjælpefunktioner"""
import numpy as np
from data_sources import get_fx_rate


def make_price_box(label, value, currency, color, sublabel="", show_secondary=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        primary_str = "-"
        secondary_div = ""
    else:
        primary_str = f"{value:,.2f} {currency}"
        if show_secondary and currency != "DKK":
            rate = get_fx_rate(currency, "DKK")
            secondary_val = value * rate
            secondary_div = (
                f"<div style='font-size:0.7rem;opacity:0.8;color:#00d4aa'>"
                f"≈ {secondary_val:,.2f} DKK</div>"
            )
        else:
            secondary_div = ""
    return (
        f"<div style='padding:0.8rem;border-radius:8px;background:{color}22;"
        f"border:1px solid {color}'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{label}</div>"
        f"<div style='font-size:1.1rem;font-weight:700'>{primary_str}</div>"
        f"{secondary_div}"
        f"<div style='font-size:0.7rem;opacity:0.6'>{sublabel}</div></div>"
    )


def make_range_box(label, low, high, currency, color, sublabel="", show_secondary=True):
    primary_str = f"{low:,.2f} - {high:,.2f} {currency}"
    if show_secondary and currency != "DKK":
        rate = get_fx_rate(currency, "DKK")
        secondary_div = (
            f"<div style='font-size:0.7rem;opacity:0.8;color:#00d4aa'>"
            f"≈ {low*rate:,.2f} - {high*rate:,.2f} DKK</div>"
        )
    else:
        secondary_div = ""
    return (
        f"<div style='padding:0.8rem;border-radius:8px;background:{color}22;"
        f"border:1px solid {color}'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{label}</div>"
        f"<div style='font-size:1.0rem;font-weight:700'>{primary_str}</div>"
        f"{secondary_div}"
        f"<div style='font-size:0.7rem;opacity:0.6'>{sublabel}</div></div>"
    )


def make_recommendation_card(label, sublabel, recommendation_text, color, score):
    return (
        f"<div style='padding:1.2rem;border-radius:12px;background:{color}22;"
        f"border:2px solid {color}'>"
        f"<div style='font-size:0.85rem;opacity:0.8'>{label}</div>"
        f"<div style='font-size:0.75rem;opacity:0.6;margin-bottom:0.5rem'>{sublabel}</div>"
        f"<div style='font-size:1.6rem;font-weight:800;color:{color}'>{recommendation_text}</div>"
        f"<div style='font-size:1.3rem;font-weight:700;margin-top:0.3rem'>{score:.0f}/100</div>"
        f"</div>"
    )
