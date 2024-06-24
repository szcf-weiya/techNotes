---
comments: true
---

# Deploy R Shiny Server on Yale SpinUp system

## Step 1: Pick a Linux server

The SpinUp platform is very similar (if not exact) to the AWS. The first thing is choose a server. I take the most familar one: Ubuntu 22.04.

## Step 2: Install R Shiny Server

Follow the official instructions: <https://posit.co/download/shiny-server/>

```bash
sudo apt-get install r-base
sudo su - -c "R -e \"install.packages('shiny', repos='https://cran.rstudio.com/')\""
sudo apt-get install gdebi-core
wget https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-1.5.22.1017-amd64.deb
sudo gdebi shiny-server-1.5.22.1017-amd64.deb
```

## Step 3: Install RStudio server (optional)

For ease of debugging, I also install the rstudio server

```bash
wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.04.2-764-amd64.deb
sudo gdebi rstudio-server-2024.04.2-764-amd64.deb
```

## Step 4: Configuration

!!! tip "`run_as`"
    If we want to use the R package installed by the current user, the best way is to set `run_as` in `/etc/shiny-server/shiny-server.conf` as the current user.

!!! tip "`preserve_logs`"
    To help debug, add `preserve_logs true;` in `/etc/shiny-server/shiny-server.conf`. But remember to comment it when it is ready for production.