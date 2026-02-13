"""
Plantilla WSGI para PythonAnywhere.

Uso:
1) Copia este contenido en el archivo WSGI que te da PythonAnywhere.
2) Ajusta PROJECT_HOME a tu ruta real en PythonAnywhere.
3) Recarga la web app desde el panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajustar esta ruta en PythonAnywhere
PROJECT_HOME = Path("/home/tu_usuario/tu_repositorio")

if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))

from app.app import app as application  # noqa: E402

