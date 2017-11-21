setwd('/Users/bauera/Dropbox/UofT/experiments/event-seg_tones/analysis')

library(plyr); library(reshape); library(lme4); library(lmerTest)

# load separate datasets and combine
df_c1_orig <- read.csv(file="forLMER_rugrats_prepped.csv", head=TRUE, sep=",")
df_c2_orig <- read.csv(file="forLMER_busyWorld_prepped.csv", head=TRUE, sep=",")
df<-rbind(df_c1_orig, df_c2_orig)

# transform data -- RT not normally distributed
# NOTE: NEG RT VALUES FROM DETRENDING -- ADD CONSTANT
df$pos_RT<-df$RT - (min(df$RT[is.finite(df$RT)]) - 1)
df$log_RT<-log(df$pos_RT)

# center and rescale some data
df<-ddply(df,c('subj'),transform,dist_cent=(dist-mean(dist)))
df<-ddply(df,c('subj'),transform,peak_cent=(peak-mean(peak)))
#df<-ddply(df,c('subj'),transform,RT_cent=scale(RT))
#df<-ddply(df,c('itemID'),transform,ave_acc=(mean(phase2_acc)))

# split into cartoon dfs
df_c1 <- subset(df, cartoon==0)
df_c2 <- subset(df, cartoon==1)

# run LMER models on RT
# cartoon 1
RT_model_c1_1 <- lmer(RT ~ dist_cent + peak_cent + afterPeak + before1stBound + aftrIncorr + chngResp + (dist_cent + peak_cent + afterPeak + before1stBound + aftrIncorr + chngResp || subj), data=df_c1)
#RT_model_c1_1 <- lmer(RT ~ dist_cent + afterPeak + (dist_cent + afterPeak || subj), data=df_c1)
summary(RT_model_c1_1)

# cartoon 2
RT_model_c2_1 <- lmer(RT ~ dist_cent + peak_cent + afterPeak + before1stBound + aftrIncorr + chngResp + (dist_cent + peak_cent + afterPeak + before1stBound + aftrIncorr + chngResp || subj), data=df_c2)
#RT_model_c2_1 <- lmer(RT ~ dist_cent + afterPeak + (dist_cent + afterPeak || subj), data=df_c2)
summary(RT_model_c2_1)

# both cartoons
#RT_model_1 <- lmer(RT ~ dist_cent + peak + afterPeak + before1stBound + cartoon + (dist_cent + peak + afterPeak + before1stBound + cartoon || subj), data=df)
#RT_model_1 <- lmer(RT ~ dist_cent + afterPeak + (dist_cent + afterPeak || subj), data=df)
#summary(RT_model_1)