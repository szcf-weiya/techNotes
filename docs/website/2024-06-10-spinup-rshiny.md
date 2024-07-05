---
comments: true
---

# Deploy Shiny Website on Yale SpinUp

This is a full record for building a shiny server  from scratch for interactive data visulaization.

## Step 1: Install R Shiny Server

1. Pick a Linux server

The SpinUp platform is very similar (if not exact) to the AWS. The first thing is choose a server. I take the most familar one: Ubuntu 22.04.

2. Install R Shiny Server

Follow the official instructions: <https://posit.co/download/shiny-server/>

```bash
sudo apt-get install r-base
sudo su - -c "R -e \"install.packages('shiny', repos='https://cran.rstudio.com/')\""
sudo apt-get install gdebi-core
wget https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-1.5.22.1017-amd64.deb
sudo gdebi shiny-server-1.5.22.1017-amd64.deb
```

3. Install RStudio server (optional)

For ease of debugging, I also install the rstudio server

```bash
wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.04.2-764-amd64.deb
sudo gdebi rstudio-server-2024.04.2-764-amd64.deb
```

4. Configuration

!!! tip "`run_as`"
    If we want to use the R package installed by the current user, the best way is to set `run_as` in `/etc/shiny-server/shiny-server.conf` as the current user.

!!! tip "`preserve_logs`"
    To help debug, add `preserve_logs true;` in `/etc/shiny-server/shiny-server.conf`. But remember to comment it when it is ready for production.

## Step 2: Write ui.R and server.R

Actually, both ui.R and server.R are simple. But there are plenty work to speed up the response time for a more friendly use.

Briefly, there are two steps: read data and plot data. Here are attempts along the time to speed up.


### Speed up Plot

1. `readRDS + Seurat::SpatialPlot`: this is the first attempt, but it turns out that the plot function from Seurat is too slow, particularly when it loads the background image, which is not necessary. 
2. disable the default behavior of loading background image in `Seurat::SpatialPlot`

```diff
$ git log
commit d9f09de15ddf05fe89a8b16eaa100e3720ee122b (HEAD, tag: v4.4.0)
$ git diff
diff --git a/R/visualization.R b/R/visualization.R
index d3edd0f4..6c48552d 100644
--- a/R/visualization.R
+++ b/R/visualization.R
@@ -6878,9 +6878,6 @@ GeomSpatial <- ggproto(
       height = unit(x = hgth, units = "npc"),
       just = c("left", "bottom")
     )
-    img.grob <- GetImage(object = image)
-
-    img <- editGrob(grob = img.grob, vp = vp)
     # spot.size <- slot(object = image, name = "spot.radius")
     spot.size <- Radius(object = image)
     coords <- coord$transform(data, panel_scales)
@@ -6897,6 +6894,9 @@ GeomSpatial <- ggproto(
     vp <- viewport()
     gt <- gTree(vp = vp)
     if (image.alpha > 0) {
+      img.grob <- GetImage(object = image)
+
+      img <- editGrob(grob = img.grob, vp = vp)
       if (image.alpha != 1) {
         img$raster = as.raster(
           x = matrix(
```

but the improvment of speed is limited.

3. take out the `ggplot` function from `SpatialPlot`, and directly use `ggplot2` without loading `Seurat`, but it is still too slow. Also tried the parallel computing for multiple plots, but actually the bottleneck is the final rendering step, which seems not for parallel computing, while the plot step itself is quite cheap. In other words, it is easy to run `p = ggplot()`, but it takes a long time to `print(p)`:

```r
> system.time({p = SpatialFeaturePlot(spatial.atac, features = feature, pt.size.factor = 1.2,
+                    image.alpha = 0, stroke = 0, alpha = c(1, 1), slot = "scale.data")})
   user  system elapsed 
  0.121   0.004   0.125 
> system.time({print(p)})
   user  system elapsed 
 24.436   4.122  28.541 


> system.time({ggsave("testp.pdf")})
Saving 11.7 x 5.45 in image
   user  system elapsed 
 24.775   4.042  28.798 
```

4. replace `ggplot2` with `plotly`, it seems faster, but here is gap between `ggplot2` and `plotly`: no direct correspondence of the custome scale in `plotly`, so it takes some time to figure out the mechanism of custom color scale. See [:link:](../R/plot.md) for more details.

### Speed up Data Loading

5. remove slots and no compress in `.rds`: since the `.rds` file is too large, it takes quite a long time to load it. The first attempt is to use remove unnecessary slots in the seurat object, and save into rds with `compress = F`, which can (significantly) shorten the data loading time.

```r
> system.time({readRDS("X_nocompress.rds")})
   user  system elapsed 
  3.839   1.906   5.741 
> system.time({readRDS("X.rds")})
   user  system elapsed 
 17.402   1.622  19.013 
```

6. use cache: to avoid repeating loading the same data, use `cache`, but note that only `cachem::cache_mem` can work, since `cachem::cache_disk` is equivalent to saving to a ".rds" file.

```bash
cm <- cachem::cache_mem(max_size = 20 * 1024^3, max_age = 60 * 60)

load_data <- memoise(function(file_path) {
  readRDS(file_path)
}, cache = cm)
```

the cache requires the RAM size, but it is expensive to increase the RAM size and it will be a waste when there is no people visiting the website.

7. convert to HDF5: since it is not necessary to load the whole data into RAM, then I save the data into `.h5` file with `writeHDF5Array`, and then load it with `HDF5Array` in `server.R`. 
   
```r
> system.time({obj_h5 = HDF5Array(paste0(filename, ".h5"), name = "data")})
   user  system elapsed 
  0.039   0.008   0.047 
> system.time({obj = readRDS(paste0(filename, ".rds"))})
   user  system elapsed 
  3.304   1.406   4.706 
```

And I encountered two tricky things:

!!! tip "use row vector instead of column vector"
    I found that the loading of `.h5` is quite slow for a column vector, but it is OK using a column vector.

!!! tip "use different name for different types of data"
    do not merge two different types of data into a data frame and save it to `.h5` since it will lose the data type


### Speed up Long Selection

The number of gene list is around 2w, the default `selectizeInput` can only accept 2000 options. Although we can specify the `maxOptions`, it will be very slow to loading the drop menu. 

```r
selectizeInput("feature", "Gene:", choices = lst_features, options = list(maxOptions = 24029))
```

The first attempt is to load the whole list on the server side, but there is a delay when selecting from the drop menu.

```r
selectizeInput("feature", "Gene:", choices = NULL) 
updateSelectizeInput(session, "feature", choices = lst_features, server = TRUE, options = list(maxOptions = 24029))
```

So I tried to adaptively update the list on the client side. However, it seems quite tricky to handle the reactive experssion. Hopefully, it succeeds. Although ChatGPT can help draft some code but it can repeatedly give the wrong answer if I asked it to debug. The tricky thing is to set `server = FALSE` in `updateSelectizeInput`, otherwise, the choices will not be updated after you deleting your selection.

??? tip "a minimal work example for adaptive loading"

    ```r
    library(shiny)

    # Generate a long list of size 20000
    long_list <- paste("Item", 1:20000)

    ui <- fluidPage(
    titlePanel("Adaptive Loading with SelectizeInput"),
    sidebarLayout(
        sidebarPanel(
        selectizeInput(
            inputId = "dynamic_select",
            label = "Select an Item",
            choices = NULL,
            options = list(
            maxOptions = 100,  # Maximum number of options to display at once
            load = I("function(query, callback) {
                if (!query.length) {
                callback();
                return;
                }
                clearTimeout(this.searchTimeout);
                var self = this;
                console.log('no time Query:', query);
                this.searchTimeout = setTimeout(function() {
                Shiny.setInputValue('query', query, {priority: 'event'});
                console.log('Query:', query);
                callback(query);
                }, 100);  // Adjust delay as needed
            }")
            )
        )
        ),
        mainPanel(
        textOutput("selected_item")
        )
    )
    )

    server <- function(input, output, session) {
    
    # Reactive to store the filtered options
    filtered_choices <- function(query) {
        if (is.null(query) || query == "") {
        return(long_list[1:100])  # Return top 100 items if query is empty
        }
        cat("Filtering choices with query:", query, "\n")
        matches <- grep(paste0("^", query), long_list, value = T, ignore.case = T)
        matches[1:min(100, length(matches))]  # Return top 100 matches
    }
    
    # Update the selectize input choices based on the query
    observeEvent(input$query, {
        cat("Received query:", input$query, "\n")
        isolate(
        {
            query = input$query
            choices <- filtered_choices(query)
        }
        )
        print(choices)
        if (!is.null(choices)) {
        cat("Updating selectize input with choices:", choices, "\n")
        session$sendCustomMessage("updateChoices", list(choices = as.list(choices)))
        updateSelectizeInput(session, "dynamic_select", choices = choices, server = FALSE)
        } else {
        cat("No choices found for query:", input$query, "\n")
        updateSelectizeInput(session, "dynamic_select", choices = character(0), server = FALSE)
        }
    })
    
    output$selected_item <- renderText({
        input$dynamic_select
    })
    }

    shinyApp(ui, server)
    ```

## Step 3: Apply for a custom domain

We wrote an email to `yalesites@yale.edu` to request a domain by simply providing the custom domain name. They replied shortly and approved our request.

## Step 4: Domain Resolution

It requires us to setup the SSL certificate. Follow the [article](https://www.digitalocean.com/community/tutorials/how-to-create-a-self-signed-ssl-certificate-for-nginx-in-ubuntu).

Also, I setup the nginx forward such that the shiny application is binded to the domain itself, i.e, no need to append the subfolder of the shiny application after the domain.

```bash
$ cat /etc/nginx/sites-enabled/shiny
    location / {
        proxy_pass http://127.0.0.1:3838/XXXX/;
```

However, after the IT set up the Application Load Balancer (ALB), we cannot access the domain, and it throws the 502 Bad Gateway error. Both IT and I struggle in the error for a quite while, later on, I found that the reason is 
that

> previously I only used "ssl_protocols TLSv1.3;" without "TLSv1.2", so after trying adding back TLSv1.2
