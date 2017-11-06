setwd('/Users/bauera/Dropbox/UofT/experiments/event-seg_tones/analysis')

library(lme4); library(lmerTest); library(reshape)

df_c1 <- read.csv(file="forLMER_rugrats_prepped.csv", head=TRUE, sep=",")
df_c2 <- read.csv(file="forLMER_busyWorld_prepped.csv", head=TRUE, sep=",")

# combine data
df<-rbind(df_c1,df_c2)

# transform data -- RT not normally distributed
# NOTE: NEG RT VALUES FROM DETRENDING -- ADD CONSTANT
df$pos_RT<-df$RT - (min(df$RT[is.finite(df$RT)]) - 1)
df$log_RT<-log(df$pos_RT)

# center and rescale some data
df<-ddply(df,c('subj'),transform,dist_cent=(dist-mean(dist)))
#df<-ddply(df,c('subj'),transform,RT_cent=scale(RT))
#df<-ddply(df,c('itemID'),transform,ave_acc=(mean(phase2_acc)))

# run LMER models on RT
# both cartoons
RT_model_1 <- lmer(log_RT ~ dist + peak + afterPeak + b41stBound + cartoon + (dist + peak + afterPeak + b41stBound + cartoon || subj), data=df)
summary(RT_model_1)

RT_model_2 <- lmer(RT ~ dist_cent + peak + afterPeak + b41stBound + cartoon + (dist_cent + peak + afterPeak + b41stBound + cartoon || subj), data=df)
summary(RT_model_2)

RT_model_3 <- lmer(log_RT ~ dist_cent + peak + afterPeak + b41stBound + cartoon + (dist_cent + peak + afterPeak + b41stBound + cartoon || subj), data=df)
summary(RT_model_3)

#cartoon 1

#cartoon 2



# BELOW: FROM OTHER STUDY ==========================================================================
RT_model_2 <- lmer((phase2_RT) ~ boundaryScoreProp * distanceSecCent + norm_RTCent + norm_acc + cartoon + (boundaryScoreProp * distanceSecCent + norm_RTCent + norm_acc + cartoon || subjID), data=df)
summary(RT_model_2)

RT_model_c1 <- lmer(log_phase2_RT ~ boundaryScoreProp + distanceSecCent + norm_RTCent + norm_acc + (boundaryScoreProp + distanceSecCent + norm_RTCent + norm_acc || subjID), data=df, subset = (cartoon==1))
RT_model_c2 <- lmer(log_phase2_RT ~ boundaryScoreProp + distanceSecCent + norm_RTCent + norm_acc + (boundaryScoreProp + distanceSecCent + norm_RTCent + norm_acc || subjID), data=df, subset = (cartoon==2))
summary(RT_model_c1)
summary(RT_model_c2)

acc_model <- glmer(phase2_acc ~ boundaryScoreProp + norm_acc + distanceSecCent + norm_RTCent+ cartoon + (boundaryScoreProp+norm_acc + distanceSecCent + norm_RTCent + cartoon || subjID), data=df, family="binomial")#subset=(ave_acc<1&ave_acc>.5))#, subset=(norm_acc<.92))
summary(acc_model)

# other analysis
RT.bound <- cast(df, subjID ~ boundaryScoreBin4, mean, value=('phase2_RT'), na.rm=T)
colMeans(RT.bound, na.rm=T)
