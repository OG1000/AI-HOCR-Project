# AI-HOCR-Project

## Brief Despcription

This is a Simple Handwriting Recognition Program. This model gives the user 6 options for different Models for OCR. It also preprocesses the images and corrects the text. I have made this program for the __SDG Goal 4 : Quality education__. This program is not fully developed and I have some [Future Goals](#future-goals).

To run, first run the __prerequisite.batch__ file then the __main4copy copy.py__ file.

## Problem Scoping
The following is the Problem Statement Template

| |  |  |
| --- |------------| --- |
| __Our__ | __Stakeholders(s) :__ Students and teachers | __Who__ |
| __has/have a problem that__ | __Issue, Problem, Need :__ There are mainly two issues that I want to solve. First is  that students know the concepts but are not able to execute in the exam, which is causing low marks for the student. The second is that the teachers while checking papers, have to go through the same answers many times a day, due to which the teachers get stresed.In both the problems, there are many other side effects. | __What__ |
| __when/while__ | __Context, Situation :__ Before and During Exams, During checking of Papers | __Where__ |
| __An ideal solution__ | __Benefit of solution for them :__ This program or AI model will help the stakeholders. This program will take a image of written text and analyse it to provide the text. It will also give the keywords. It will help the students by helping them to write the correct answers with apt keywords, which is the main cause of marks. It will help teachers to give the keywords for fast checking. | __Why__ |

## Data Acquisition
I have used the dataset __[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)__ for training the models. 


## Data Exploration
As the above datasets are readymade datasets, Data exploration was not required. The required data was already structured and formated before hand. 

## Modelling

### OCR
I have given 7 options (1 created by me) for the user with Square brackets rating. _S is for speed and A is for accuracy_ 
| Model Name                       | Framework/Lib      | Handwriting Support | Speed         | Accuracy      | Notes / Link                                                                 | S | A |
|---------------------------------- |--------------------|---------------------|--------------|--------------|------------------------------------------------------------------------------|--------------|-----------|
| DocTR CRNN+VGG16 | Mindee/DocTR       | Yes                | Fast         | High         | Uses doctr.models.ocr_predictor with CRNN+VGG16, good for printed & some handwritten | 8 | 7 |
| TrOCR (Base)                      | HuggingFace/Transformers | Yes           | Medium       | High         | https://huggingface.co/microsoft/trocr-base-handwritten                      | 6 | 9 |
| TrOCR (Large)                     | HuggingFace/Transformers | Yes           | Slow         | Very High    | https://huggingface.co/microsoft/trocr-large-handwritten                     | 4 | 9 |
| Donut                             | HuggingFace/Transformers | Yes           | Medium       | High         | https://huggingface.co/naver-clova-ix/donut-base                             | 5 | 8 |
| DocTR (CRNN, SAR, PARSeq, etc.)   | Mindee/DocTR       | Yes                | Fast         | High         | https://github.com/mindee/doctr                                              | - | - |
| PARSeq                            | Mindee/DocTR       | Yes                | Fast         | High         | https://github.com/mindee/doctr                                              | 7 | 8 |
| SAR                               | Mindee/DocTR/MMOCR | Yes                | Fast         | High         | https://github.com/mindee/doctr                                              | 8 | 8 |
| EMNIST_CNN (My Model) | PyTorch  | Yes  | Fast | Medium | Trained in [Google Colab](colab.research.google.com) | 9 | 6 |
|   EMNIST_CNN Model (My Model) | PyTorch | Yes | Fast | High | Trained in [Google Colab](colab.research.google.com) | 9 | 7 |

__Note :__ For the training code, use - [HOCR Model Training.ipynb](https://colab.research.google.com/drive/1uOd9cIHQ2zInzSqMNEcH51vEV0AQBuhL?usp=sharing) 

__*__ The speed and accuracy rating may be misleading, a model can perform better based on the image. These ratings are relative. 
## Evaluation
In the above table, the speed and accuracy ratings are the evaluations. 

In the above table, the speed and accuracy ratings are the evaluations.
In my model, it is trained till 90% Accuracy (104840/116323 Test Batches), with average loss of 0.26954
 
## Deployment

### GUI
This program has a very simple GUI. I have used _tkinter_ module which is in-built in python. But, I am planning to use _customtkinter_ which is __not__ in-built in python and has to be installed. `pip install customtkinter` 

### Preprocessing
For preprocessing I have used the PIL library ( `pip install pillow` ) 

## Future Goals
- Improving the accuracy of the model or developing a model for Handwriting OCR with better accuracy

- Adding a keyword extractor and write-up checker for giving marks for the text written

- Integrating this program with a main homescreen and improving the GUI

- Improving the minimum requirement to let this program run in old systems

- Improving the start up time

- Distributing  the program for worldwide use

