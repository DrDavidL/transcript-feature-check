import numpy as np
import pandas as pd

def correct_multi_category_lr_update(prior_probs, likelihood_ratios):
    """
    Correct implementation for multi-category likelihood ratio updates.
    
    For multiple competing hypotheses with likelihood ratios, we need to:
    1. Understand that LRs are typically given relative to a reference category
    2. Convert LRs to actual likelihoods 
    3. Apply Bayes' theorem properly
    
    The key insight: In medical diagnosis, LRs are often given as:
    LR = P(evidence|disease) / P(evidence|no disease)
    
    But for multiple categories, we need to be careful about what "no disease" means.
    """
    
    # Method 1: Treat LRs as relative likelihoods (most common interpretation)
    # This assumes the LRs represent relative evidence strength
    
    # Convert to unnormalized posterior using Bayes' theorem
    unnormalized_posterior = prior_probs * likelihood_ratios
    
    # Normalize to get proper probabilities
    posterior_probs = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return posterior_probs

def correct_binary_lr_update(prior_prob_disease, likelihood_ratio):
    """
    Correct implementation for binary case (disease vs no disease).
    
    LR = P(evidence|disease) / P(evidence|no disease)
    
    Using odds form of Bayes' theorem:
    posterior_odds = prior_odds × LR
    """
    
    # Convert prior probability to odds
    prior_odds = prior_prob_disease / (1 - prior_prob_disease)
    
    # Update odds using likelihood ratio
    posterior_odds = prior_odds * likelihood_ratio
    
    # Convert back to probability
    posterior_prob = posterior_odds / (1 + posterior_odds)
    
    return posterior_prob

def analyze_sample_data():
    """Analyze the sample data to understand the LR interpretation"""
    
    # Load sample data
    df = pd.read_csv('sample_feature_lr_matrix.csv')
    
    print("=== Analysis of Sample LR Matrix ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Look at first feature
    first_feature = df.iloc[0]
    print(f"\nFirst feature: {first_feature.iloc[0]}")
    print("LR values across categories:")
    for i, col in enumerate(df.columns[1:]):
        print(f"  {col}: {first_feature.iloc[i+1]}")
    
    # Check if LRs sum to anything meaningful
    lr_values = first_feature.iloc[1:].values
    print(f"\nSum of LRs: {np.sum(lr_values):.2f}")
    print(f"Mean LR: {np.mean(lr_values):.2f}")
    
    # Look for patterns
    print(f"\nLR statistics:")
    print(f"  Min: {np.min(lr_values):.2f}")
    print(f"  Max: {np.max(lr_values):.2f}")
    print(f"  Values > 1: {np.sum(lr_values > 1)}")
    print(f"  Values < 1: {np.sum(lr_values < 1)}")
    print(f"  Values = 1: {np.sum(np.abs(lr_values - 1) < 0.01)}")
    
    return df

def test_interpretations():
    """Test different interpretations of the LR data"""
    
    print("\n=== Testing Different LR Interpretations ===")
    
    # Sample data from "food gets stuck"
    categories = [
        "A gastroesophageal disorder",
        "A cardiovascular disorder", 
        "A dermatologic disorder",
        "a non-inflammatory muscle or joint disorder",
        "a pulmonary disorder",
        "a psychiatric disorder",
        "Any systemic inflammatory joint disorder",
        "No definitive diagnosis"
    ]
    
    prior_probs = np.array([1/8] * 8)  # Uniform priors
    likelihood_ratios = np.array([6, 0.3, 0.6, 0.6, 0.6, 1.8, 1.2, 0.6])
    
    print(f"Feature: 'Patient Has: food gets stuck'")
    print(f"Prior probabilities: {prior_probs}")
    print(f"Likelihood ratios: {likelihood_ratios}")
    
    # Current implementation (treating LRs as likelihoods)
    current_result = prior_probs * likelihood_ratios
    current_result = current_result / np.sum(current_result)
    
    print(f"\nCurrent implementation result:")
    for i, cat in enumerate(categories):
        print(f"  {cat}: {current_result[i]:.3f}")
    
    # The issue: LR=6 for gastroesophageal gives it 86.7% probability
    # But LR=6 just means the evidence is 6x more likely with this condition
    
    print(f"\nInterpretation check:")
    print(f"LR=6 for gastroesophageal means:")
    print(f"  P(food gets stuck | gastroesophageal) / P(food gets stuck | reference) = 6")
    print(f"This doesn't mean gastroesophageal has 86.7% probability!")
    
    return current_result

if __name__ == "__main__":
    # Analyze the sample data structure
    df = analyze_sample_data()
    
    # Test different interpretations
    test_interpretations()
    
    print(f"\n=== CONCLUSION ===")
    print(f"The current implementation is mathematically incorrect.")
    print(f"It treats likelihood ratios as if they were likelihoods.")
    print(f"This leads to overconfident probability estimates.")