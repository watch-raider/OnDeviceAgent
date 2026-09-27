# set base image (host OS)
FROM python:3.12-slim

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/main.py src/trading_agent.py src/finance_tools.py src/constants.py ./

# command to run on container start
CMD ["fastapi", "run", "./main.py", "--port", "80"]