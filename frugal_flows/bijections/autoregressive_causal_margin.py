#This is the new causal bijection. 
# It maps (Y_1, ..., Y_K) → (R_1, ..., R_K) via the autoregressive Rosenblatt transform conditioned on T. 
# It is an AbstractBijection of shape (K,) with cond_shape = (1,) (the treatment).

# Location-translation case (exact ATE specification):
# forward (Y → R):   R_k = F(Y_k - τ_k * T | Y_{<k})   [CDF of residual given parents]
# inverse (R → Y):   Y_k = F^{-1}(R_k | Y_{<k}) + τ_k * T
# where F(· | Y_{<k}) is itself a small normalizing flow (an inner autoregressive flow over the K pixels conditioned on the previous pixels). τ ∈ R^K is the vector ATE — trainable, readable after fitting.

# This class is the counterpart of UnivariateNormalCDF for scalar FF. It wraps:

# A masked_autoregressive_flow over K dimensions conditioned on Y_{<k} and T
# A vector tau of shape (K,) for the location shift
# It is marked NonTrainable in the outer stack (same as UnivariateNormalCDF today) but tau itself is a trainable leaf — same pattern as ate in LocCond.