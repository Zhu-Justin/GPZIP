http://bitly.ws/8stA
# Heartsteps data set

# analyze the treatment effect of activity suggestion on step count using Gaussian Process

# Tianchen Qian
# 2020.04.29


rm(list = ls())

library(GauPro)
library(tidyverse)

# 0. preparation ---------------------------------------------------------

### load data ###

# loading data from mounted mBox "HeartSteps" folder
sys.var <- switch(Sys.info()["sysname"],
                  "Windows" = list(locale = "English",
                                   mbox = "Z:/HeartSteps/"),
                  "Darwin" = list(locale = "en_US",
                                  mbox = "/Volumes/dav/HeartSteps/"),
                  "Linux" = list(locale = "en_US.UTF-8",
                                 mbox = "~/mbox/HeartSteps/"))
# sys.var$mbox.data <- paste0(sys.var$mbox, "Data/")
sys.var$mbox.data <- paste0(sys.var$mbox, "Tianchen Qian/")
Sys.setlocale("LC_ALL", sys.var$locale)
Sys.setenv(TZ = "GMT")

# load(paste0(sys.var$mbox.data, "jbslot_traveldayremoved_90min.RData"))
jbslot90 <- readRDS(paste0(sys.var$mbox.data, "jbslot_traveldayremoved_90min.RDS"))

# jbslot90 <- readRDS("jbslot_traveldayremoved_90min.RDS")

jbslot90$day.in.week <- weekdays(jbslot90$decision.utime, abbr = TRUE)
jbslot90$is.weekend <- grepl("S(at|un)", jbslot90$day.in.week)

jbslot90$p.send <- 0.6
jbslot90$p.send.active <- 0.3
jbslot90$p.send.sedentary <- 0.3

jbslot90$anySteps <- as.numeric(jbslot90$steps > 0)


# total_T <- 60
total_T <- 90

dta_use <- filter(jbslot90, min.from.last.decision >= total_T, min.after.decision < total_T)
# dta_use <- filter(jbslot90, avail == TRUE, min.from.last.decision >= total_T, min.after.decision < total_T)
# Among the 6331 decisions with avail == TRUE, 10 decision points are excluded here because their min.from.last.decision < total_T.

dta_use$jbsteps30pre <- exp(dta_use$jbsteps30pre.log) - 0.5


### Use data from user 1, first 15 decision points (the available ones) ###

dta_tmp <- filter(dta_use, user == 1 & decision.index.nogap <= 15 & avail == 1)

x_mat <- cbind(rep(1, nrow(dta_tmp)), dta_tmp$send, dta_tmp$min.after.decision)

y <- dta_tmp$steps.log - mean(dta_tmp$steps.log) # centering log step y

gp <- GauPro(x_mat, y, parallel=FALSE)

newdata0 <- cbind(rep(1, total_T), 0, 0:(total_T-1))
newdata1 <- cbind(rep(1, total_T), 1, 0:(total_T-1))

predict0 <- gp$predict(newdata0)
se0 <- gp$predict(newdata0, se=T)$se
predict1 <- gp$predict(newdata1)
se1 <- gp$predict(newdata1, se=T)$se


# Make a plot
plot(dta_tmp$min.after.decision[dta_tmp$send == 0], y[dta_tmp$send == 0], type = "p", col = "blue",
     ylab = "centered log step count", xlab = "min.after.decision", ylim = c(-5, 5),
     main = "red: treatment. blue: no treatment. dashed line: confidence interval.")
points(dta_tmp$min.after.decision[dta_tmp$send == 1], y[dta_tmp$send == 1], col = "red")
lines(0:(total_T-1), predict0, col = "blue")
lines(0:(total_T-1), predict0 + 1.96 * se0, col = "blue", lty = 2)
lines(0:(total_T-1), predict0 - 1.96 * se0, col = "blue", lty = 2)
lines(0:(total_T-1), predict1, col = "red")
lines(0:(total_T-1), predict1 + 1.96 * se1, col = "red", lty = 2)
lines(0:(total_T-1), predict1 - 1.96 * se1, col = "red", lty = 2)

