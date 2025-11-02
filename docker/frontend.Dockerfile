FROM node:20-alpine as build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./frontend/
COPY frontend ./frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

FROM nginx:stable-alpine
COPY --from=build /app/frontend/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
