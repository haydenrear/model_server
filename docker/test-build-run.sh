#!/bin/zsh

./build.sh
docker build . -t ms
./after-build.sh
docker-compose up -d

sleep 10

curl -X POST http://localhost:9991/ai_suite_gemini_embedding --data '{}' -H 'Content-Type: application/json'

docker-compose down
