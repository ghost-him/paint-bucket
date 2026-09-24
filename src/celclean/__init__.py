"""celclean - remove AI-generation noise from cel-shaded (flat colour) artwork.

Public API::

    from celclean import clean, Options
    out, info = clean(rgba_uint8, Options(strength=1.0))
"""

from .pipeline import Options, auto_radius, clean

__version__ = "0.1.0"
__all__ = ["clean", "Options", "auto_radius", "__version__"]
