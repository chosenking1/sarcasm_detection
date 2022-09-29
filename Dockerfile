
# Builds a considerably smaller image
FROM python:3.8.13-slim-bullseye
# # set work directory

# COPY ./app /app
WORKDIR /app

RUN pip install --upgrade setuptools

# copy project
COPY . .

#RUN #pip -r requirements.txt

RUN pip install -r requirements.txt


EXPOSE 8000:8000
# command to run on container start
#--host 0.0.0.0
CMD ["uvicorn", "app:app", "--reload ", "--host", "127.0.0.2", "--port", "8000"]

