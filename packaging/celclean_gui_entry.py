"""Entry script for the frozen build: PyInstaller wants a file, not a module.

``celclean/gui/__main__.py`` covers the ``python -m`` route; this one is referenced by
``packaging/celclean-gui.spec``.
"""

from celclean.gui.app import main

raise SystemExit(main())
