# Test error analysis

- Images: 297
- Accuracy: 0.7609
- Macro-F1: 0.7617
- Incorrect: 71
- Review threshold: 0.80; review count: 95
- Accuracy on accepted predictions: 0.905940594059406

## Most frequent confusions
- book -> notebook: 9 image(s)
- stapler -> chair: 5 image(s)
- phone -> notebook: 4 image(s)
- bin -> stapler: 3 image(s)
- book -> chair: 3 image(s)
- notebook -> book: 3 image(s)

## Manual error review (student to complete)
- Examine incorrect photographs listed in predictions.csv.
- Record actual causes: glare, cropping, clutter, angle, lighting, class similarity or label mistakes.
- Identify at least 3 representative images when errors exist; explain corrective data collection.
- Do not modify the model using test data; changes require a new sealed test set.
