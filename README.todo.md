- Modify the prediction method to support multiple body shapes (3 genders × 64 shapes each).

- Run inference once over the full dataset, saving predictions for every batch.

- Store all generated results for downstream inspection and analysis.

- Visualize the saved motions using a custom visualization tool to assess quality.

- Define quantitative thresholds to determine which results are acceptable.

- Select high-quality results and integrate them into the imitation learning training loop.

- check the musk, make sure every frame is valid