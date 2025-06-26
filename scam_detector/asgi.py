"""
ASGI config for scam_detector project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scam_detector.settings')

application = get_asgi_application() 