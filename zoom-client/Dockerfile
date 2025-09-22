FROM node:24-slim

WORKDIR /app

COPY package*.json ./

USER root

RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3 && \
  rm -rf /var/lib/apt/lists/*

RUN npm install

COPY . .

RUN chown -R node:node /app

USER node

EXPOSE 8080

EXPOSE 3000

CMD [ "node", "index.js" ]
