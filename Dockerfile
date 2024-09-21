FROM python:3.10
RUN mkdir /model_server_home
RUN cd /model_server_home
WORKDIR model_server_home
COPY ./src/model_server /model_server_home/model_server
COPY ./drools_py_copied /model_server_home/drools_py
COPY ./python_di_copied /model_server_home/python_di
COPY ./python_util_copied /model_server_home/python_util
COPY ./metadata_extractor_copied /model_server_home/metadata_extractor
COPY ./pasta/pasta /model_server_home/pasta
COPY ./requirements.txt /model_server_home/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY ./docker.env ./.env
RUN mkdir ./resources
COPY ./resources/application-docker.yml /model_server_home/resources/application.yml
RUN mkdir /model_server_home/work
RUN mkdir /model_server_home/test_work
RUN cd /model_server_home
EXPOSE 9991
CMD python3 -m model_server.main.model_server_main