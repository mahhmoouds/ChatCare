#!/bin/bash
# Cloudflare Pages build script

# Install dependencies
pip install -r requirements.txt

# Collect static files
cd myproject
python manage.py collectstatic --noinput

# Run migrations (if needed)
# python manage.py migrate --noinput

echo "Build completed successfully!"

