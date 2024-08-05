# Top-K LLM Ranking
TBD!

## Installation
In order to run the project, first make sure that you have `docker` and `docker compose` properly installed. Then you can run the project using following command:

```bash
docker compose up --build 
```

If you need to only run one service, check the `docker-compose.yml` to find the service name (e.g. questioner), then use following command to run individual service

```bash
docker compose up --build {service_name}
```

If you want to run the project in detach mode, you can add `-d` to previous commands. 

