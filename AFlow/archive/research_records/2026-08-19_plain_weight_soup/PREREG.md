# Fixed checkpoint weight soup preregistration

Create exactly one single-model soup: `0.75 × historical epoch168 parameters + 0.25 × EMA168 parameters`. The ratio is copied from the successful logit ensemble and is not searched. Floating tensors are linearly interpolated; integer buffers come from the historical model. The model is evaluated first on fixed validation108 with raw-logit threshold zero. If scan and subject do not both improve, the direction stops without test or additional alpha values.

This is the zero-training, one-model inference experiment motivated by Model Soups. Different training seeds may occupy different basins, so failure is expected and inexpensive.
