# Enhanced Transcript Feature Analysis with Belief Updating

This application processes clinical transcripts to assess features and perform sophisticated belief updating calculations using Bayesian inference, KL divergence, and entropy reduction analysis.

## Features

### Core Functionality
- **Feature Assessment**: Uses LLM to determine which clinical features are addressed in transcripts (binary output: 1/0)
- **Belief Updating**: Sequential Bayesian inference using likelihood ratios
- **Information Theory**: Calculates KL divergence and entropy reduction for each feature
- **Enhanced Reports**: Comprehensive DOCX reports with mathematical calculations

### Mathematical Calculations
- **Entropy Calculation**: -Σ(p * log₂(p)) using log base 2
- **KL Divergence**: Σ(q * log₂(q/p)) for belief updating
- **Sequential Processing**: Features processed in order with cumulative statistics
- **Belief Updating**: Prior × LR → normalize for posterior probabilities

## Workflow

### Step 1: Upload Feature-LR Matrix
Upload a CSV or Excel file with:
- **First column**: Clinical features (e.g., "Patient Has: food gets stuck")
- **Remaining columns**: Diagnostic categories with likelihood ratio values

### Step 2: Set Prior Probabilities
**For Multiple Diagnostic Categories**: Enter prior probabilities for each category. Values must sum to 1.0.

**For Single Diagnostic Category**: Enter the pretest probability (any positive value, e.g., 0.35 for 35%). The system will use binary Bayesian updating (target diagnosis vs. "everything else").

### Step 3: Upload Transcripts
Upload PDF transcript files for analysis.

### Step 4: Generate Reports
The system will:
1. Extract text from PDFs
2. Use LLM to assess which features are addressed (1/0)
3. Apply likelihood ratios for addressed features
4. Perform sequential belief updating
5. Calculate KL divergence and entropy reduction
6. Generate enhanced DOCX reports

## Output Reports

Each DOCX report includes:

### 1. Feature Assessment Summary
- Total features assessed
- Features addressed vs. not addressed
- Percentage addressed

### 2. Complete Feature Assessment
- All features with binary results (1/0)

### 3. Addressed Features with Likelihood Ratios
- Features that were addressed
- LR values for each diagnostic category

### 4. Belief Updating Sequence
- Step-by-step probability updates
- Entropy before/after each feature
- Entropy reduction per feature
- KL divergence per feature

### 5. Final Diagnostic Probabilities
- Updated probabilities for each diagnostic category

### 6. Summary Statistics
- Initial and final entropy
- Total entropy reduction
- Cumulative KL divergence
- Number of belief updates performed

## Sample Data

The repository includes:
- `sample_feature_lr_matrix.csv`: Example feature-LR matrix with 78 clinical features
- `sample_transcript.txt`: Example clinical transcript for testing

## Requirements

```
streamlit
pandas
PyPDF2
python-docx
openpyxl
openai
numpy
```

## Installation

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up secrets in `.streamlit/secrets.toml`:
   ```toml
   PASSWORD="your-password"
   OPENROUTER_API_KEY="your-openrouter-api-key"
   ```
4. Run the app: `streamlit run app.py`

## Mathematical Background

### Belief Updating Process

**For Multiple Diagnostic Categories:**
1. **Prior Probabilities**: Initial diagnostic category probabilities (π) - must sum to 1.0
2. **Likelihood Ratios**: Evidence strength for each feature (LR)
3. **Unnormalized Posterior**: π × LR
4. **Normalization**: Divide by sum to ensure probabilities sum to 1
5. **Posterior Probabilities**: Updated beliefs (q)

**For Single Diagnostic Category:**
1. **Pretest Probability**: Initial probability of the target diagnosis (any positive value)
2. **Likelihood Ratios**: Evidence strength for each feature (LR)
3. **Odds Conversion**: Convert pretest probability to odds
4. **Odds Updating**: posterior_odds = prior_odds × LR
5. **Probability Conversion**: Convert back to probability using binary Bayesian updating

### Mathematical Implementation Details

**For Multiple Categories:**
```
posterior ∝ prior × LR_value
posterior = (prior × LR) / Σ(prior × LR)
```

**For Single Category (Binary Updating):**
```
prior_odds = pretest_prob / (1 - pretest_prob)
posterior_odds = prior_odds × LR
posterior_prob = posterior_odds / (1 + posterior_odds)
```

**Important Notes:**
- **Multiple categories**: LR values treated as relative evidence weights
- **Single category**: Uses proper binary Bayesian updating with odds form
- Values > 1 indicate evidence supporting the diagnosis
- Values < 1 indicate evidence against the diagnosis
- Values = 1 indicate neutral evidence

**For Manual Verification:**

**Multiple Categories:**
1. Multiply each prior probability by its corresponding LR value
2. Sum all products to get the normalization constant
3. Divide each product by the normalization constant
4. Results should match the application output

**Single Category:**
1. Convert pretest probability to odds: odds = prob / (1 - prob)
2. Multiply odds by LR: new_odds = odds × LR
3. Convert back to probability: new_prob = new_odds / (1 + new_odds)

**Example Calculations:**

**Multiple Categories** - Feature "Patient Has: food gets stuck" with uniform priors (0.125):
- Gastroesophageal: 0.125 × 6.0 = 0.75
- Cardiovascular: 0.125 × 0.3 = 0.0375
- Sum all products = 1.4625
- Final probability = 0.75 / 1.4625 = 0.513 (51.3%)

**Single Category** - 35% pretest probability with LR = 6.0:
- Prior odds: 0.35 / (1 - 0.35) = 0.538
- Posterior odds: 0.538 × 6.0 = 3.23
- Posterior probability: 3.23 / (1 + 3.23) = 0.764 (76.4%)

### Information Theory Metrics
- **Entropy**: Measures uncertainty in probability distribution
- **KL Divergence**: Measures information gained from belief updating
- **Entropy Reduction**: Decrease in uncertainty after evidence

### Sequential Processing
Features are processed one at a time, with each posterior becoming the prior for the next feature, allowing cumulative belief updating throughout the clinical interview.

## Supported LLM Models

- Google Gemini 2.5 Flash/Pro
- OpenAI GPT-4.1/GPT-4.1-mini
- Anthropic Claude 3.7 Sonnet

## Error Handling

The application includes robust error handling for:
- Invalid file formats
- Mathematical edge cases (log of zero)
- Prior probability validation (different rules for single vs. multiple categories)
- LLM response parsing
- File processing errors

## Use Cases

- Medical education assessment
- Clinical reasoning analysis
- Diagnostic probability tracking
- Information theory research
- Bayesian inference demonstration