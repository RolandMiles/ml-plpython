# используем официальный образ postgres
FROM postgres:13

# устанавливаем переменные окружения для postgres
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=mldb

# устанавливаем python, pip (для установки пакетов), venv (для создания окружения),
# plpython (для выполнения python скриптов в pg), libpq-dev (для установки psycopg2)
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv postgresql-plpython3-$PG_MAJOR && \
    apt-get install -y libpq-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# создаем виртуальное окружение python
RUN python3 -m venv /opt/venv

# копируем файл со списком необходимых для установки библиотек
COPY requirements.txt /opt/venv

# устанавливаем в виртуальное окружение необходимые библиотеки
RUN /opt/venv/bin/pip install -r /opt/venv/requirements.txt 

# копируем сырые данные обучающей и проверочной выборок (имитация загрузки от источника)
COPY raw_train_data.csv raw_test_data.csv /home/ml/data/

# копируем ddl-скрипты в директорию, где они будут автоматически выполнены при запуске контейнера
COPY init.sql /docker-entrypoint-initdb.d/

# устанавливаем переменную окружения для использования виртуального окружения по умолчанию
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"