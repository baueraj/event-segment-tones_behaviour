# https://www.statmethods.net/stats/power.html

library(pwr)

pwr.t.test(n=100, d = 0.533, sig.level = 0.05, type = c("two.sample", "one.sample", "paired"))