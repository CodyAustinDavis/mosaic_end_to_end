# mosaic_end_to_end
Example of a Mosaic + Databricks end to end data engineering, llm fine tuning, and model serving demo on UC. 

This demo ingests multiple raw data sources, cleans, standardizes, and combines them into a standard training data set. The demo then uses that data set governed and tracked by Unity Catalog to fine-tune an MPT model with Mosaic reading from UC Volumes where MDS datasets are saved. Finally, the demo shows the registration of the fine-tuned model with UC via ML Flow and served via REST API on the new optimized Databricks Model Serving platform. 


## Steps