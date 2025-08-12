import numpy as np
import pandas as pd

def current_incorrect_implementation(prior_probs, likelihood_ratios):
    """Current implementation - INCORRECT"""
    unnormalized_posterior = prior_probs * likelihood_ratios
    normalization_constant = np.sum(unnormalized_posterior)
    posterior_probs = unnormalized_posterior / normalization_constant
    return posterior_probs, normalization_constant

def correct_implementation(prior_probs, likelihood_ratios):
    """Correct implementation using proper Bayesian inference with LRs"""
    # For likelihood ratios, we need to use the odds form of Bayes' theorem
    # posterior_odds = prior_odds × LR
    
    # Convert prior probabilities to odds
    prior_odds = prior_probs / (1 - prior_probs)
    
    # Update odds using likelihood ratios
    posterior_odds = prior_odds * likelihood_ratios
    
    # Convert back to probabilities
    posterior_probs = posterior_odds / (1 + posterior_odds)
    
    # Normalize to ensure they sum to 1 (for multiple competing hypotheses)
    posterior_probs = posterior_probs / np.sum(posterior_probs)
    
    return posterior_probs

def test_simple_case():
    """Test with a simple 2-category case"""
    print("=== Testing Simple 2-Category Case ===")
    
    # Simple case: 2 diagnostic categories
    prior_probs = np.array([0.3, 0.7])  # Prior probabilities
    likelihood_ratios = np.array([2.0, 0.5])  # LR for each category
    
    print(f"Prior probabilities: {prior_probs}")
    print(f"Likelihood ratios: {likelihood_ratios}")
    
    # Current incorrect implementation
    incorrect_result, norm_const = current_incorrect_implementation(prior_probs, likelihood_ratios)
    print(f"\nCurrent (INCORRECT) implementation:")
    print(f"Posterior probabilities: {incorrect_result}")
    print(f"Sum: {np.sum(incorrect_result)}")
    
    # Manual calculation for verification
    print(f"\nManual calculation:")
    print(f"Prior odds: {prior_probs[0]/(1-prior_probs[0]):.3f} : {prior_probs[1]/(1-prior_probs[1]):.3f}")
    
    # For binary case, LR should be applied to convert prior odds to posterior odds
    prior_odds_ratio = (prior_probs[0]/(1-prior_probs[0])) / (prior_probs[1]/(1-prior_probs[1]))
    posterior_odds_ratio = prior_odds_ratio * (likelihood_ratios[0] / likelihood_ratios[1])
    
    # Convert back to probabilities
    manual_posterior_0 = posterior_odds_ratio / (1 + posterior_odds_ratio)
    manual_posterior_1 = 1 / (1 + posterior_odds_ratio)
    
    print(f"Manual posterior probabilities: [{manual_posterior_0:.4f}, {manual_posterior_1:.4f}]")
    print(f"Manual sum: {manual_posterior_0 + manual_posterior_1}")

def test_multi_category_case():
    """Test with multiple categories like in the sample data"""
    print("\n=== Testing Multi-Category Case ===")
    
    # Use actual data from sample
    categories = ["A gastroesophageal disorder", "A cardiovascular disorder", "A dermatologic disorder"]
    prior_probs = np.array([0.33, 0.33, 0.34])  # Roughly equal priors
    likelihood_ratios = np.array([6.0, 0.3, 0.6])  # From "food gets stuck" feature
    
    print(f"Categories: {categories}")
    print(f"Prior probabilities: {prior_probs}")
    print(f"Likelihood ratios: {likelihood_ratios}")
    
    # Current incorrect implementation
    incorrect_result, norm_const = current_incorrect_implementation(prior_probs, likelihood_ratios)
    print(f"\nCurrent (INCORRECT) implementation:")
    print(f"Posterior probabilities: {incorrect_result}")
    print(f"Sum: {np.sum(incorrect_result)}")
    
    print(f"\nThe issue: Current implementation treats LRs as likelihoods")
    print(f"This gives gastroesophageal disorder {incorrect_result[0]:.1%} probability")
    print(f"But LR=6.0 means evidence is 6x more likely given gastroesophageal vs baseline")

if __name__ == "__main__":
    test_simple_case()
    test_multi_category_case()