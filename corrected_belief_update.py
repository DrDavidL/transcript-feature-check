import numpy as np
from typing import Tuple, Dict

def perform_belief_update_current(prior_probs: np.ndarray, likelihood_ratios: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Current implementation - treats LRs as relative evidence weights.
    This is what the existing code does.
    """
    unnormalized_posterior = prior_probs * likelihood_ratios
    normalization_constant = np.sum(unnormalized_posterior)
    posterior_probs = unnormalized_posterior / normalization_constant
    return posterior_probs, normalization_constant

def perform_belief_update_correct_binary(prior_prob: float, likelihood_ratio: float) -> float:
    """
    Mathematically correct implementation for binary case (disease vs no disease).
    
    Uses the odds form of Bayes' theorem:
    posterior_odds = prior_odds × LR
    
    Where LR = P(evidence|disease) / P(evidence|no disease)
    """
    # Convert prior probability to odds
    prior_odds = prior_prob / (1 - prior_prob)
    
    # Update odds using likelihood ratio
    posterior_odds = prior_odds * likelihood_ratio
    
    # Convert back to probability
    posterior_prob = posterior_odds / (1 + posterior_odds)
    
    return posterior_prob

def perform_belief_update_correct_multi_category(prior_probs: np.ndarray, 
                                                likelihood_ratios: np.ndarray,
                                                reference_category_idx: int = -1) -> Tuple[np.ndarray, float]:
    """
    Mathematically correct implementation for multi-category case.
    
    Assumes likelihood ratios are given relative to a reference category
    (typically the last category like "No definitive diagnosis").
    
    For each category i: LR_i = P(evidence|category_i) / P(evidence|reference)
    
    This converts LRs back to relative likelihoods and applies Bayes' theorem correctly.
    """
    
    # Extract reference category probability and LR
    ref_prior = prior_probs[reference_category_idx]
    ref_lr = likelihood_ratios[reference_category_idx]
    
    # For the reference category, LR should be 1.0 by definition
    # If it's not, we need to normalize all LRs
    if not np.isclose(ref_lr, 1.0, atol=0.1):
        # Normalize all LRs relative to the reference
        normalized_lrs = likelihood_ratios / ref_lr
    else:
        normalized_lrs = likelihood_ratios.copy()
    
    # Convert LRs to relative likelihoods
    # If LR_i = P(evidence|category_i) / P(evidence|reference), then:
    # P(evidence|category_i) = LR_i × P(evidence|reference)
    
    # We can set P(evidence|reference) = 1 for relative calculations
    relative_likelihoods = normalized_lrs
    
    # Apply Bayes' theorem: posterior ∝ prior × likelihood
    unnormalized_posterior = prior_probs * relative_likelihoods
    
    # Normalize
    normalization_constant = np.sum(unnormalized_posterior)
    posterior_probs = unnormalized_posterior / normalization_constant
    
    return posterior_probs, normalization_constant

def compare_implementations():
    """Compare the different implementations with sample data."""
    
    print("=== Comparison of Belief Update Implementations ===\n")
    
    # Sample data from "food gets stuck" feature
    categories = [
        "Gastroesophageal", "Cardiovascular", "Dermatologic", 
        "Muscle/Joint", "Pulmonary", "Psychiatric", 
        "Inflammatory Joint", "No Diagnosis"
    ]
    
    prior_probs = np.array([0.125] * 8)  # Uniform priors
    likelihood_ratios = np.array([6.0, 0.3, 0.6, 0.6, 0.6, 1.8, 1.2, 0.6])
    
    print(f"Feature: 'Patient Has: food gets stuck'")
    print(f"Prior probabilities: {prior_probs}")
    print(f"Likelihood ratios: {likelihood_ratios}\n")
    
    # Current implementation
    current_result, current_norm = perform_belief_update_current(prior_probs, likelihood_ratios)
    
    print("1. CURRENT IMPLEMENTATION (treats LRs as evidence weights):")
    for i, cat in enumerate(categories):
        print(f"   {cat}: {current_result[i]:.3f}")
    print(f"   Normalization constant: {current_norm:.3f}\n")
    
    # Corrected multi-category implementation
    corrected_result, corrected_norm = perform_belief_update_correct_multi_category(
        prior_probs, likelihood_ratios, reference_category_idx=-1
    )
    
    print("2. CORRECTED IMPLEMENTATION (proper LR interpretation):")
    for i, cat in enumerate(categories):
        print(f"   {cat}: {corrected_result[i]:.3f}")
    print(f"   Normalization constant: {corrected_norm:.3f}\n")
    
    # Binary example for gastroesophageal vs all others
    print("3. BINARY EXAMPLE (Gastroesophageal vs Others):")
    gastro_prior = 0.125
    gastro_lr = 6.0
    gastro_posterior = perform_belief_update_correct_binary(gastro_prior, gastro_lr)
    print(f"   Prior P(Gastroesophageal): {gastro_prior:.3f}")
    print(f"   Likelihood Ratio: {gastro_lr}")
    print(f"   Posterior P(Gastroesophageal): {gastro_posterior:.3f}")
    print(f"   Posterior P(Others): {1-gastro_posterior:.3f}\n")
    
    # Analysis
    print("=== ANALYSIS ===")
    print("The current implementation gives gastroesophageal disorder")
    print(f"a probability of {current_result[0]:.1%}, which seems reasonable")
    print("given that LR=6.0 indicates strong evidence.")
    print()
    print("However, the mathematical interpretation depends on how")
    print("the likelihood ratios were originally calculated.")
    print()
    print("If they represent true statistical likelihood ratios")
    print("relative to a baseline, the corrected implementation")
    print("should be used.")

if __name__ == "__main__":
    compare_implementations()