# Spatiotemporal Wildlife Image Classification
Bachelor's thesis investigating **if** and **how** observation **spatiotemporal metadata** can be used to improve 
existing **wildlife image classification** processes. 

# Development
## Image Classification
The image classification models executes model training and validation from two separate Docker images. 
This allows for easy model training and validation using a GPU unit, on the same image as originally run. 
Any code changes to any of the image classification files do not require the images to be rebuilt as the project directory is 
copied into the container on execution. Any code changes are automatically picked up by the Docker container. 

### Download Docker Images
The docker images are packages in this repository, to download the latest train and validate packages execute the following: 
```angular2html
docker pull docker.pkg.github.com/trav-d13/spatiotemporal_wildlife_classification/image_train:latest
docker pull docker.pkg.github.com/trav-d13/spatiotemporal_wildlife_classification/image_validate:latest
```

### Train New Model
1. Specify the name of the model and the path to the appropriate image directory in `src/models/image/taxonomic_modelling.py`
   - The documentation provides examples. 
2. In the terminal please execute the following command to train the CNN using an available GPU unit.

```angular2html
docker run --gpus all -u $(id -u):$(id -g) -v /path/to/project/root:/app/ -w /app -t ghcr.io/trav-d13/spatiotemporal_wildlife_classification/train_image:latest
```
You will see information updating you on the training process printed to terminal/

### Validate New Model




# Information Summary
Automated wildlife classification is essential within ecological studies, wildlife conservation and management, 
specifically fulfilling the roles of species population estimates, individual identification, and behavioural patterns.

Observation metadata includes features such as location, time of observation, date, and multiple environmental features 
gathered from [open-meteo](https://open-meteo.com/). For a full list of features please review: ..... 

A novel approach combines metadata classification and existing wildlife classification methodologies, within the 
taxonomic hierarchical structure, to accurate classify images down to the sub-species taxonomic level.


More information on the proposed methods can be found within the [Methods](#methods) 
section and [Thesis document](#external-links).

##### Proposed use-case
A proposed use-case utilizes tourist's public social media postings in order to support wildlife park's population 
estimates and tracking. 
Two considerations are kept in mind. Firstly, this information must be kept confidential in order to eliminate 
the threat of granting poachers additional knowledge. 
Secondly, the immediacy of social media posting and the staggering quantity of historical resources make it 
the largest potential source of wildlife historical and current data.

## Issues in Wildlife Image Classification
#### Modelling 
Image classification relies on labelled images, forming the dataset. 
However, most wildlife image datasets are comprised of images of **varying quality**, and quantity, 
creating **unbalanced** datasets, where some species have very few observations. 
Current wildlife classification processes experience a decreasing classification performance going down 
the taxonomic tree. 

#### Data Subject
The data subject itself, wildlife species, make the classification process difficult, largely due to similar looking 
species, and sub-species, in addition to sympatric species (species found in the same geographic area). 
`
| ![](http://static.inaturalist.org/photos/88383/medium.jpg) | ![](https://inaturalist-open-data.s3.amazonaws.com/photos/9581740/medium.jpg) |
|------------------------------------------------------------|-------------------------------------------------------------------------------|
| Panthera pardus                                            | Panthera Onca                                                                 |`

## Data
The training data is obtained from [iNaturalist](https://www.inaturalist.org/), a citizen-science based platform tasked with generating global research-grade, annotated flora and fauna images to facilitate computer vision developement. 
This thesis focuses exclusively on a subgroup of mammalian families, Elephantidae and Felids. 
The below images serve as sample images from [iNaturalist](https://www.inaturalist.org/)

The link to the used dataset is provided [here]() (inactive)

| ![](https://inaturalist-open-data.s3.amazonaws.com/photos/254323960/large.jpeg) | ![](https://inaturalist-open-data.s3.amazonaws.com/photos/254318111/large.jpeg) |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| ![](https://inaturalist-open-data.s3.amazonaws.com/photos/254306053/large.jpg) | ![](https://static.inaturalist.org/photos/254074172/large.jpg)                                                                           |


### Data Characteristics
##### Geographic Distribution
![](resources/readme_resources/geographic_distribution.png)

##### Example Precipitation Profile for Elephantidae Family
![](resources/readme_resources/elephant_precipitation_profile.png)

For a further look at basic exploratory dataset analysis please preview the available `notebooks`

## Methods
The novel approach, makes use of a dual taxonomic tree approach. 
The taxonomic trees will be specialized within image classification, and metadata classification respectively. 
At each node of each tree, the node will contain a classifier, capable of classifying the children nodes, and restricted
by its parent node. The nodes at each respective taxonomic level are capable of communicating, in order to effectively
agree on a classification, leaning into their respective strengths.

![](resources/readme_resources/dual_dt.png)


## Results
Still to be completed

## External Links
- **Thesis Document** (inactive)
- [iNaturalist](https://www.inaturalist.org/)

### Scientific Papers
