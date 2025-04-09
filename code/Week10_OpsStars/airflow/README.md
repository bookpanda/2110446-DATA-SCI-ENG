```bash
airflow db init

airflow users create \
    --username admin \
    --firstname yourname \
    --lastname yourname \
    --role Admin \
    --email your@email.com \
    --password yourpassword

airflow scheduler
airflow webserver --port 8080

```