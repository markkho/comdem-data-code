library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

library(nlme)
library(lme4)
library(lmerTest)

#=============================================#
#   Logistic Regression on correct responses (contrast coding)  #
#=============================================#
responses <- read.csv("../data/exp1L-trials.csv")
contrasts(responses$learner_inst) <- c(-0.5, 0.5)
contrasts(responses$teacher_inst) <- c(-0.5, 0.5)

fullModel <- glmer(
  correct ~ learner_inst*teacher_inst + (1 | rf) + (1 | teacher_id) + (1 | workerid),
  data = responses,
  family = binomial
)
summary(fullModel)

#=============================================#
#   Linear Regression on confidence responses (contrast coding)  #
#=============================================#

fullModel <- lmer(
  confidence ~ teacher_inst*learner_inst + (1 | rf) + (1 | teacher_id) + (1 | workerid),
  data = responses,
  REML = FALSE
)

summary(fullModel)