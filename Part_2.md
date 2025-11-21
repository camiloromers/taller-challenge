# taller-challenge

## 1. Define “Success” in Business Terms
A model is successful if it reduces total financial loss from returns by:
Catching as many true returns as possible while avoiding unnecessary interventions on non-returns
The final business objective will be: `Minimize total financial cost = return losses + intervention losses`

## 2. Recommended Metrics (Business-Aligned)
A good set of metrics will be:
- 1. Recall (class 1) — “Catch rate” of costly returns
False Negatives are very expensive.
High recall reduces missed returns and It will help to save money.
- 2. Precision (class 1)
A false positive intervention costs $3.
Low precision will be translated to wasted interventions and operational losses.
- 3. Precision-Recall AUC:
Better for imbalanced settings datasets.

## 3. False Positives vs False Negatives (Trade-Off Analysis)
The cost of the false negative is -18 USD (return predicted as no return)
**This is the worst possible error**.
And the cost for a false positive is -3 USD which is a cheaper mistake (non-return predicted as return)
So the optimal model will:
- Accept more false positives
- Increase recall even if precision drops

## 4. Calculate financial impact of predictions

```python
def compute_financial_impact(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    
    impact = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1 and yt == 1:
            impact += 15
        elif yp == 1 and yt == 0:
            impact -= 3
        elif yp == 0 and yt == 1:
            impact -= 18    
    return impact
```

the result is: -9090 USD

## 5. Determine optimal threshold
```python
thresholds = np.linspace(0.05, 0.95, 10)
results = []

for thre in thresholds:
    impact = compute_financial_impact(y_test, y_proba, threshold=thre)
    results.append((thre, impact))

df = pd.DataFrame(results, columns=['threshold', 'total_impact'])
df.sort_values('total_impact', ascending=False).head()
```

I tried to compute the impact for all thresholds and the value is always -9090 which means we always predict `0` class.

## 6. Critical Questions
Because missing a return (–$18) is 6× more expensive than wasting an intervention (–$3), the optimal balance leans heavily toward recall even if precision decreases.
The target balance is maximize recall of class 1 up to the point where additional false positives (precision drop) begin to reduce net financial benefit.
