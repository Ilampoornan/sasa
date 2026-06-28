# Group 1916 Temporal Analysis (PCA-Style)

SAE: checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10
Model: gpt2 | Hook: blocks.7.hook_resid_pre
Group: 1751 | Group rank: 6
Representation: latent | min_group_norm: 0.0
Corpus aliases: True
Plot source: all

## Coverage
- days: n=35, mean_group_norm=11.7776, std_group_norm=0.3324, mean_vec_norm=11.7776, std_vec_norm=0.3324
- months: n=60, mean_group_norm=11.4322, std_group_norm=0.2587, mean_vec_norm=11.4322, std_vec_norm=0.2587
- years: n=220, mean_group_norm=11.7618, std_group_norm=0.3578, mean_vec_norm=11.7618, std_vec_norm=0.3578

## PCA diagnostics
- days: EVR=[0.426, 0.316, 0.184, 0.044, 0.025], best_plane=PC4 vs PC5, order_alignment=0.397, ring_cv=0.405
- months: EVR=[0.767, 0.107, 0.098, 0.018, 0.007], best_plane=PC4 vs PC5, order_alignment=0.371, ring_cv=0.567
- years: EVR=[0.596, 0.214, 0.100, 0.053, 0.024], best_pc=PC2, spearman(|year, pc|)=0.270

## PCA scatter planes (MDF-style)
- days: plane=PC4-PC5
- months: plane=PC4-PC5
- years: plane=PC2-PC1

## PCA plane diagnostics (sequential pairs)
- days:
  - PC1-PC2: order_alignment=0.063, ring_cv=0.274
  - PC2-PC3: order_alignment=0.093, ring_cv=0.307
  - PC3-PC4: order_alignment=0.109, ring_cv=0.394
  - PC4-PC5: order_alignment=0.397, ring_cv=0.405
- months:
  - PC1-PC2: order_alignment=0.029, ring_cv=0.496
  - PC2-PC3: order_alignment=0.109, ring_cv=0.363
  - PC3-PC4: order_alignment=0.216, ring_cv=0.311
  - PC4-PC5: order_alignment=0.371, ring_cv=0.567

## Embedding comparisons (plot source)
- days: best=Isomap (order=0.076, ring_cv=0.210); PCA-best=PC4-PC5 (order=0.397, ring_cv=0.405)
- months: best=Isomap (order=0.099, ring_cv=0.569); PCA-best=PC4-PC5 (order=0.371, ring_cv=0.567)

## Cone check (PC1 vs radius in PC2–PC3)
- days: pc1_radius_corr=0.643
- months: pc1_radius_corr=-0.120

## Order-aligned projection (fit to circular labels; metrics on label means)
- days: order_alignment=0.841, ring_cv=0.337
- months: order_alignment=0.700, ring_cv=0.661

## Qualitative interpretation (aligned with 2405.14860v3)
- The figure uses an order-aligned 2D projection (ridge fit to circular labels) to make the cyclic structure for days/months explicit; PCA plane diagnostics above quantify how well PCA axes capture the circle.
- Years are evaluated with the most monotonic PCA axis (Spearman), highlighting linear temporal structure rather than a strong circular ordering.
- Main ring plots are normalized to a unit circle (angles only), explicitly ignoring radial magnitude to emphasize cyclic ordering; embedding comparisons show raw vs circle-projected points.
- MDF-style PCA scatter plots mirror Fig. 1 of 2405.14860v3 by showing the PCA plane with the clearest cyclic structure (days/months) or strongest year gradient.

## Top decoder tokens (group direction)
' nonetheless', ':', ':-', ':(', ' cautiously', ' nevertheless', ' again', '.', ' thanks', ' additionally', ' insofar', ' however'

## Top activating prompts
### days
- norm=12.4086 | term=Tuesday | The meeting is on Tuesday.
- norm=12.4063 | term=Monday | The meeting is on Monday.
- norm=12.3831 | term=Thursday | The meeting is on Thursday.
- norm=12.3283 | term=Wednesday | The meeting is on Wednesday.
- norm=12.2890 | term=Sunday | The meeting is on Sunday.

### months
- norm=11.9829 | term=September | The festival is held in September.
- norm=11.9725 | term=January | The festival is held in January.
- norm=11.8842 | term=February | The festival is held in February.
- norm=11.8639 | term=April | The festival is held in April.
- norm=11.8491 | term=July | The festival is held in July.

### years
- norm=12.6269 | term=2013 | The company was founded in 2013.
- norm=12.5793 | term=2014 | The company was founded in 2014.
- norm=12.5413 | term=2012 | The company was founded in 2012.
- norm=12.5093 | term=2015 | The company was founded in 2015.
- norm=12.4990 | term=2013 | The treaty was signed in 2013.
