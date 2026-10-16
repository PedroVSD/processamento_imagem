Work on digital pathology. The objective is to investigate, in a reproducible manner, how choices regarding preprocessing, representation, classification, aggregation, and validation affect the robustness of the system.

- classification
- clustering

As a starting point, cluster by colors and then apply a CNN

### 2. Initial glossary

|Term|Brief description|
|---|---|
|**Baseline**|Simple and reproducible implementation used as a point of comparison.|
|**Calibration**|Correspondence between the confidence stated by the model and how often it is correct.|
|**Domain**|Data acquisition context, such as laboratory, scanner, or staining protocol.|
|**H&E**|Hematoxylin and eosin staining, common in histopathological images.|
|**Digital slide / WSI**|Digital image of an entire slide (_Whole Slide Image_), potentially gigapixels in size.|
|**Domain shift**|Change in the image distribution between training and use, caused by origin, staining, or acquisition.|
|**Patch / tile**|Smaller crop extracted from an image or slide, processable individually.|
|**Patient/slide split**|Separation that prevents images from the same patient or slide from appearing in both training and testing.|
|**External validation**|Evaluation on data collected from a different source than the one used for training.|

### 3. Learning objectives

- Apply image processing operations to a real problem, justifying the choices made.
- Analyze color, illumination, texture, noise, resolution, focus, and acquisition artifacts.
- Build and compare [image classification](https://www.dcc.ufrrj.br/moodle/mod/resource/view.php?id=17638 "Classificação de Imagens") baselines.
- Organize experiments with proper separation between development and evaluation.
- Measure predictive quality, uncertainty, computational cost, and robustness.
- Communicate limitations, biases, and risks of a computer vision application in healthcare.
