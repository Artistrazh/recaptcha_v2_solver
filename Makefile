up:
	docker-compose up --build -d

down:
	docker-compose down

in:
	docker exec -it captcha_service bash

go:
	docker exec -i captcha_service /bin/bash -c "python3 /app/src/run.py"

solver:
	docker exec -i captcha_service /bin/bash -c "python3 /app/Solver/main.py"
