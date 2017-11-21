# https://www.statmethods.net/stats/power.html

library(pwr)

pwr.t.test(d = 0.533, sig.level = 0.05, power = 0.8, type = c("two.sample", "one.sample", "paired"))