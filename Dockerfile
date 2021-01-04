FROM huentelemu/nlp:latest
ENV PYTHONUNBUFFERED 1

LABEL maintainer="Pablo Huentelemu <pablo.huentelemu@gmail.com>"

# Create rootless user for security reasons
RUN useradd -ms /bin/bash user
WORKDIR /home/user

# Create django app folder
RUN mkdir django
COPY ./django django
WORKDIR django

# Copy model
COPY ./model .

# Allow the execution of the entrypoint
RUN chmod +x docker_entrypoint_local.sh

# Swap to rootless user
USER user

# Execute django
CMD ["/home/user/django/docker_entrypoint_local.sh"]
