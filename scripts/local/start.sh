#! /bin/bash
#
# Examples:
#
# To start up containers (always rebuilding images)
# 
# ./scripts/start.sh
#

# cp .env from env_template if it doesn't exist
if [ -f ".env" ]; then
    echo "Found .env"
else
  echo "Cannot find .env. Copying from env_template"
  cp env_template .env
fi

# Always rebuild the images
echo "Building new images from compose"
docker-compose -f docker/docker-compose.yml --project-name merf_dev up -d --build
