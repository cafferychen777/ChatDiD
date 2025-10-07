# ChatDiD Example Datasets

This directory contains example datasets for testing and demonstrating ChatDiD's difference-in-differences analysis capabilities. These datasets cover various DID scenarios from classic papers to modern staggered adoption designs.

## 📊 Available Datasets

### 1. **mpdta.csv** - Callaway & Sant'Anna (2021)
**Source**: `did` R package example data  
**Description**: County-level teen employment data with staggered minimum wage adoption  
**Use Case**: Modern heterogeneity-robust DID estimation

**Variables:**
- `year`: Time period (2003-2007)
- `countyreal`: County identifier
- `lpop`: Log population
- `lemp`: Log employment (outcome)
- `first.treat`: First treatment year (cohort identifier)
- `treat`: Treatment indicator

**Analysis Type**: Staggered DID with multiple cohorts  
**Recommended Estimator**: Callaway & Sant'Anna (`did::att_gt`)

---

### 2. **card_krueger_1994.csv** - Card & Krueger (1994) Style
**Source**: Simulated based on classic minimum wage study  
**Description**: Fast-food employment before/after NJ minimum wage increase  
**Use Case**: Classic 2×2 DID design

**Variables:**
- `store_id`: Restaurant identifier
- `state`: NJ (treated) or PA (control)
- `chain`: Restaurant chain
- `fte_before/after`: Full-time equivalent employees
- `wage_before/after`: Starting wage
- `co_owned`, `centralj`, `southj`, `pa1`, `pa2`: Controls

**Analysis Type**: Simple 2×2 DID  
**Recommended Estimator**: Standard TWFE or robust methods for comparison

---

### 3. **card_krueger_1994_long.csv** - Long Format Version
**Source**: Same as above, reshaped to long format  
**Description**: Panel format of Card-Krueger data  
**Use Case**: Panel data DID analysis

**Variables:**
- `store_id`: Restaurant identifier
- `state`: NJ (treated) or PA (control)
- `period`: "before" or "after"
- `year`, `month`: Time identifiers
- `fte`: Full-time equivalent employees (outcome)
- `wage`: Starting wage
- `treated`: Treatment group indicator (1=NJ, 0=PA)
- `post`: Post-treatment period indicator

**Analysis Type**: Panel DID  
**Recommended Estimator**: TWFE with robust standard errors

---

### 4. **staggered_adoption.csv** - Modern Staggered DID
**Source**: Simulated based on modern DID literature  
**Description**: County-level data with staggered policy adoption  
**Use Case**: Testing heterogeneity-robust estimators

**Variables:**
- `county`: County identifier (1-100)
- `year`: Time period (2000-2010)
- `lemp`: Log employment (outcome)
- `lpop`: Log population (control)
- `first_treat`: First treatment year (0=never treated)
- `gname`: Group name for CS estimator
- `treat`: Current treatment status

**Analysis Type**: Staggered DID with multiple cohorts  
**Recommended Estimator**: Callaway & Sant'Anna, Sun & Abraham, or imputation methods

---

### 5. **simple_2x2_did.csv** - Basic DID Example
**Source**: Simulated for teaching purposes  
**Description**: Simple 2×2 DID with known treatment effect  
**Use Case**: Learning and testing basic DID concepts

**Variables:**
- `unit`: Unit identifier (1-50)
- `time`: Time period (0=pre, 1=post)
- `treated`: Treatment group (1=treated, 0=control)
- `post`: Post-treatment indicator
- `outcome`: Outcome variable (true effect = 4)
- `x1`, `x2`: Control variables

**Analysis Type**: Basic 2×2 DID  
**True Treatment Effect**: 4.0  
**Recommended Estimator**: Standard TWFE (should work well)

---

## 🚀 Quick Start Examples

### Load Data in ChatDiD
```python
# Load any dataset
load_data("data/examples/mpdta.csv")

# Explore the data structure
explore_data()

# Run diagnostics
diagnose_twfe(
    unit_col="countyreal", 
    time_col="year", 
    outcome_col="lemp", 
    treatment_col="treat"
)
```

### Recommended Analysis Workflows

#### For **mpdta.csv** (Staggered DID):
1. Load data → Explore → Diagnose TWFE bias
2. Use Callaway & Sant'Anna estimator
3. Check parallel trends with power analysis
4. Conduct sensitivity analysis

#### For **card_krueger_1994.csv** (Classic DID):
1. Load data → Explore → Basic DID analysis
2. Compare TWFE vs robust estimators
3. Validate with simple difference-in-means

#### For **staggered_adoption.csv** (Modern DID):
1. Load data → Explore → Diagnose TWFE bias
2. Compare multiple robust estimators
3. Event study analysis
4. Sensitivity testing

## 📚 References

1. **Card, D., & Krueger, A. B. (1994)**. Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772-793.

2. **Callaway, B., & Sant'Anna, P. H. (2021)**. Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

3. **Goodman-Bacon, A. (2021)**. Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

4. **Roth, J., Sant'Anna, P. H., Bilinski, A., & Poe, J. (2023)**. What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics*, 235(2), 2218-2244.

## 💡 Tips for Analysis

- **Always start with data exploration** to understand your panel structure
- **Run TWFE diagnostics** before choosing an estimator
- **Use multiple robust estimators** for comparison
- **Check parallel trends assumption** with formal tests
- **Conduct sensitivity analysis** to test robustness

For more detailed analysis examples, see the ChatDiD documentation and tutorials.
