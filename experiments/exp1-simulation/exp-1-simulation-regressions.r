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

#Analyze human teacher effects on simulated observers
exp_traj_beliefs <- read.csv("./_comps/exp_traj_beliefs.csv")
btraj_model <- lmer(
  final_btarg ~ (1 | participant) + (1 | mdp_code) + cond,
  data = exp_traj_beliefs
)
summary(btraj_model)
lmer_fixedeffects_summary(btraj_model, 'condshow')

b2traj_model <- lmer(
  final_b2targ ~ (1 | participant) + (1 | mdp_code) + cond,
  data = exp_traj_beliefs
)
summary(b2traj_model)
lmer_fixedeffects_summary(b2traj_model, 'condshow')

b2jtraj_model <- lmer(
  final_b2jtarg ~ (1 | participant) + (1 | mdp_code) + cond,
  data = exp_traj_beliefs
)
summary(b2jtraj_model)
lmer_fixedeffects_summary(b2jtraj_model, 'condshow')

b2jcomtraj_model <- lmer(
  b2j_com ~ (1 | participant) + (1 | mdp_code) + cond,
  data = exp_traj_beliefs
)
summary(b2jcomtraj_model)
lmer_fixedeffects_summary(b2jcomtraj_model, 'condshow')

#Analyze statistics of teacher behavior - entropy
traj_stats <- read.csv("./_comps/full_beh_mod.csv")

entmodel <- lmer(
  unknown_color_ent_exp ~ unknown_color_ent_sim
      + (1 | participant) + (1 | mdp_code),
  data = traj_stats
)
summary(entmodel)

#Analyze statistics of teacher behavior - color visitation
mod_color_visit <- read.csv('./_comps/mod_color_visit.csv')
mod_color_visit$color_tile = mod_color_visit$color_tile == 'True'

colorvisit_mod <- glmer(
  color_tile ~ model_uncertain_prop + (1 | participant) + (1 | rf),
  data = mod_color_visit,
  family=binomial,
  control=glmerControl(optimizer="bobyqa",
                       optCtrl=list(maxfun=3e5))
)
summary(colorvisit_mod)