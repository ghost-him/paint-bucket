"""celclean - remove AI-generation noise from cel-shaded (flat colour) artwork.

Public API::

    from celclean import clean, Options
    out, info = clean(rgba_uint8, Options(strength=1.0))
    opaque = flatten(rgba_uint8, parse_color("white"))
"""

from .pipeline import Options, auto_radius, clean, flatten, parse_color

__version__ = "0.1.0"
__all__ = ["clean", "flatten", "Options", "auto_radius", "parse_color", "__version__"]
