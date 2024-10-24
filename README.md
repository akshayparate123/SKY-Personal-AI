# SKY-Personal-AI

This repository will not function as expected if you simply download it, as many large files are missing and cannot be uploaded here. Please note that this is a licensed repository, and it is protected by law. Unauthorized downloading or use of this repository, whether for commercial or personal purposes, may lead to legal consequences.

## Introduction to SKY Personal Assistant

The SKY Personal Assistant is a state-of-the-art virtual assistant designed to enhance productivity and simplify everyday tasks. This AI-driven assistant is capable of handling a variety of functions, ranging from answering questions to retrieving information from the internet. It can perform simple tasks like writing Python code and summarizing content. Moreover, SKY Personal Assistant has the ability to create detailed user profiles, enabling personalized responses in future interactions. It continuously learns and evolves through a robust feedback system. In future implementations, it will incorporate advanced features like computer vision and data analysis to provide even more comprehensive assistance.

***Core Features of SKY Personal Assistant***

![png](https://github.com/akshayparate123/SKY-Personal-AI/blob/main/Images/Features_1.png)

***Future Implementations and Enhancements***
![png](https://github.com/akshayparate123/SKY-Personal-AI/blob/main/Images/FutureImplementations.png)

## Datasets Used

***Emotions***

```python
dataset_1 = load_dataset("ma2za/many_emotions",trust_remote_code=True)
dataset_2 = load_dataset("Villian7/Emotions_Data")
dataset_3 = load_dataset("Villian7/Emotions_Data")
```

***QA***
```python
dataset_1 = load_dataset("virattt/financial-qa-10K")
dataset_2 = load_dataset("DeividasM/financial-instruction-aq22")
dataset_3 = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
dataset_4 = load_dataset("meta-math/MetaMathQA")
dataset_5 = load_dataset("yahma/alpaca-cleaned")
dataset_6 = load_dataset("benjaminbeilharz/better_daily_dialog")
dataset_7 = load_dataset("li2017dailydialog/daily_dialog",trust_remote_code=True)
dataset_8 = load_dataset("Isotonic/human_assistant_conversation_deduped")
```

***Title***
```python
dataset_1 = load_dataset("Ateeqq/news-title-generator")
dataset_2 = load_dataset("dreamproit/bill_summary_us")
```

***Keywords***
```python
dataset_1 = load_dataset("Ateeqq/Title-Keywords-SEO")
```


***Summary***
```python
dataset_1 = load_dataset("dreamproit/bill_summary_us")
```

## Pre-Trained Model Used (BART)
![png](https://github.com/akshayparate123/SKY-Personal-AI/blob/main/Images/BART_arch.png)
## RAG Pipeline

![png](https://github.com/akshayparate123/SKY-Personal-AI/blob/main/Images/rag.png)

**Conclusion**

The SKY Personal Assistant represents a significant advancement in virtual assistant technology. Its current capabilities, including question answering, information retrieval, task performance, summarization, user profiling, and continuous learning, make it an invaluable tool for users. The planned future enhancements, such as computer vision integration, advanced data analysis, and improved user interactions, promise to take its functionality to even greater heights. SKY is not just an assistant but a continuously evolving partner designed to make life easier and more efficient for its users. With SKY, the future of personal assistance is here, and it’s brighter than ever.




