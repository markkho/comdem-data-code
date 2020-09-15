library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

library(nlme)
library(lme4)
library(lmerTest)

trials <- read.csv("../data/exp2b-responses.csv")

#contrast coding
contrasts(trials$observer_cond) <- c(-0.5, 0.5)
contrasts(trials$demonstrator_cond) <- c(-0.5, 0.5)

#=============================================#
#   Logistic Regression on correct responses  #
#=============================================#
teachLearn_M <- glmer(
  correct ~ observer_cond*demonstrator_cond + 
    (trialnum | tf:grid) + (1 | teacher_id) + (1 | learner_id),
  data = trials,
  family = binomial
)
summary(teachLearn_M)

#==============================================#
#   Linear Regression on confidence responses  #
#==============================================#
teachLearn_M <- lmer(
  confidence ~ observer_cond*demonstrator_cond + 
    (trialnum | tf:grid) + (1 | teacher_id) + (trialnum | learner_id),
  data = trials,
  REML=TRUE,
  control = lmerControl(
    optimizer="bobyqa",
    optCtrl=list(maxfun=20000)
  )
)
summary(teachLearn_M)