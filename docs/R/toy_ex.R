# Sample data
df <- read.table(header=T, text='
                 cond xval yval
                 A    1  2.0
                 A    2  2.5
                 B    1  3.0
                 B    2  2.0
                 ')
library(ggplot2)

# Plot with standard lines and points
# group = cond tells it which points to connect with lines
ggplot(df, aes(x=xval, y=yval, group = cond)) +
  geom_line() +
  geom_point()

# Set overall shapes and line type
ggplot(df, aes(x=xval, y=yval, group = cond)) +
  geom_line(linetype="dashed",  # Dashed line
            size = 1.5) +       # Thicker line
  geom_point(shape = 0,         # Hollow squares
             size = 4)          # Large points

# Condition shapes and line type on variable cond
ggplot(df, aes(x=xval, y=yval, group = cond)) +
  geom_line(aes(linetype=cond), # Line type depends on cond
            size = 1.5) +       # Thicker line
  geom_point(aes(shape=cond),   # Shape depends on cond
             size = 4)          # Large points

# Same as previous, but also change the specific linetypes and
# shapes that are used
ggplot(df, aes(x=xval, y=yval, group = cond)) +
  geom_line(aes(linetype=cond), # Line type depends on cond
            size = 1.5) +       # Thicker line
  geom_point(aes(shape=cond),   # Shape depends on cond
             size = 4) +        # Large points
  scale_shape_manual(values=c(NA,5)) +                  # Change shapes
  scale_linetype_manual(values=c(1, 0)) # Change linetypes