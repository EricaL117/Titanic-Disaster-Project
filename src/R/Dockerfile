FROM r-base:4.3.2
WORKDIR /app
COPY install_packages.R /app/install_packages.R
RUN Rscript install_packages.R
COPY . /app
CMD ["Rscript", "main.R"]
