library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

library(nlme)
library(lme4)
library(lmerTest)

# Helper functions
lmer_fixedeffects_summary <- function(model, variable_name) {
  mod_sum <- summary(model)
  pval <- mod_sum$coefficients[variable_name,"Pr(>|t|)"]
  if (pval < .001) {
    pval <- "< .001"
  } else if (pval < .01) {
    pval <- "< .01"
  } else if (pval < .05) {
    pval <- "< .05"
  } else {
    pval <- sprintf("%.2f", pval)
  }
  return(sprintf("$\beta=%.2f$, $SE=%.2f$,$t(%.2f)=%.2f$, $p %s$ [Satterthwaite's approximation]",
                 mod_sum$coefficients[variable_name,"Estimate"],
                 mod_sum$coefficients[variable_name,"Std. Error"],
                 mod_sum$coefficients[variable_name,"df"],
                 mod_sum$coefficients[variable_name,"t value"],
                 pval
  ))
}



#analysis of human teaching model
exp_traj_beliefs <- read.csv("./_comps/exp_traj_beliefs.csv")
btraj_model <- lmer(
  b.target ~ (1 | participant) + (1 | grid:tf) + cond,
  data = exp_traj_beliefs
)
summary(btraj_model)
lmer_fixedeffects_summary(btraj_model, 'condshow')

b2traj_model <- lmer(
  b2.target ~ (1 | participant) + (1 | grid:tf) + cond,
  data = exp_traj_beliefs
)
summary(b2traj_model)
lmer_fixedeffects_summary(b2traj_model, 'condshow')

b2jtraj_model <- lmer(
  b2j.target ~ (1 | participant) + (1 | grid:tf) + cond,
  data = exp_traj_beliefs
)
summary(b2jtraj_model)
lmer_fixedeffects_summary(b2jtraj_model, 'condshow')

b2jcomtraj_model <- lmer(
  b2j.com ~ (1 | participant) + (1 | grid:tf) + cond,
  data = exp_traj_beliefs
)
summary(b2jcomtraj_model)
lmer_fixedeffects_summary(b2jcomtraj_model, 'condshow')

#analysis of jumping
jump_data <- read.csv("./_comps/expsim_trials_jumps.csv")
jump_model <- glmer(
  is_jump ~ ( 1 | participant) + (1 | grid:tf) + sim_jumps_per_traj,
  data=jump_data,
  family=binomial
)
summary(jump_model)

#analysis of risky jumping
risky_jump_data <- read.csv("./_comps/expsim_trials_risky_jumps.csv")
risky_jump_model <- glmer(
  is_risky_jump ~ ( 1 | participant) + (1 | grid:tf) + sim_risky_jumps_per_jump,
  data=risky_jump_data,
  family=binomial
)
summary(risky_jump_model)
