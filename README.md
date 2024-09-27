# Gene Resistance Prediction

## Project Goal

This project aimed to develop predictive models for drug resistance testing for the drugs listed below:

- Chloroquine
- Dihydroartemisinin
- Lumefantrine
- Quinine
- Halofantrine
- and Piperaquine.

## Project Desciption

It is focused on gene resistance prediction, allowing users to upload genome sequence files, which is then preprocessed and fed into machine learning models for resistance prediction. This work utilized both software engineering and machine learning expertise to deliver a solution that could assist researchers in making informed decisions about drug resistance.

Models were built using dataset for each of the drugs provided (Chloroquine, Dihydroartemisinin, Lumefantrine, Quinine, Halofantrine, and Piperaquine). Six(6) different models were generated from the dataset representing each of the drugs. These models were then stored in pickle files with User Interface(UI) and Application Programming Interface(API) created using streamlit. The model was tested locally using files stored in _consensus_sequences_ folder through streamlit in the file _app.py_. When the required results were achieved, it was finally deployed to streamlit cloud at this [Link](https://gene-resistance-prediction-aazkmjfvmmyzd84umrstv5.streamlit.app/).

### Interacting with the Project UI

On the UI, we have the option to upload a file from our computer or enter a link to a valid genome sequence file with a _.fa_ extension. We then proceed to the dropdown provided and select the drug we want to test for. The result is outputted and there is an option to download to a csv file.

## Links, Folders and Files

### Links

- Project Interface: https://gene-resistance-prediction-aazkmjfvmmyzd84umrstv5.streamlit.app/
- GitHub Repository: https://github.com/profgreatwonder/resistance-prediction
- Documentation: https://github.com/profgreatwonder/resistance-prediction#README

### Folders and Files

- **archived**: contains unused notebooks
- **consensus_sequences**: contains the genome sequence files.
- **csv_dataset_files**: contains the csv files representing the drugs used in training the models
- **datadict_files**: contains csv files generated during model training
- **model_training_notebooks_binary_classification_random_forest**: contains notebooks used in training the different models with the dataset representing the different drugs.
- **pickle_files**: contains pickle files used in prediction stored in app.py
- **xlxs_dataset_files**: contains excel files which were converted to csv files for training
- **app.py**: contains the logic responsible for the UI that gives access for the model to be used in prediction
- **README.md**: contains the project documentation
- **requirements.txt**: contains the technologies used and their versions. Used by streamlit to deploy the app on their servers. gotten by running the command below inside of our activated environment, saving it to a ".txt" file and editing:

        pip list

- **requirements.yml**: also contains technologies used and their versions. Used to recreate the project using conda on your local machine. This file is gotten from running the command:

        conda env export --no-builds > requirements.yml

**Note 1**: _--no-builds_ makes sure that the technologies are not generated with the versions specific to the operating system used for the project.

**Note 2**: the code below present in all the notebook is important to help generate the present working directory for your machine. With it, you can replace the filepath present in the project with the right location on your local.

        import os
        print("Current working directory:", os.getcwd())

## Dataset Description

The dataset has columns:

- Accession_no
- sequence_id
- Consensus_sequence
- label

## Software and Tools

1. **VSCODE**: for managing, writing, and organizing code efficiently, with debugging and version control features.
2. **Jupyter Notebook**: used within VSCode for writing and executing Python code, visualizing data, and documenting the workflow seamlessly in one environment.
3. **Git/Github**: for version control and connecting to streamlit for deployment
4. **Python**: for developing machine learning models and data processing.
5. **Scikit-learn**: for building predictive models.
6. **Streamlit**: for deploying the application and providing an interactive interface.
7. **Pandas and NumPy**: for handling and transforming the dataset.
8. **Joblib**: for model serialization.
9. **BioPython**: for parsing genomic sequences in FASTA and FASTQ formats.

## Replicating the Project

The environment was created, activated and requirements stored in yaml file by running the following commands:

for .yml

- conda create --name ML-Project-on-Resistance-Prediction
- conda activate ML-Project-on-Resistance-Prediction
- conda env export --no-builds > environment.yml

To replicate the environment, do the following:

- Clone the Project
- Create the environment and install all requirements by running the command:

      conda env create -f environment.yml

for .txt with conda

- conda create --name ML-Project-on-Resistance-Prediction
- conda activate ML-Project-on-Resistance-Prediction
- conda list --export > requirements.txt

To replicate the environment, do the following:

- Clone the Project
- Create the environment and install all requirements by running the command:

        conda create --name <env_name> --file requirements.txt

To run the streamlit UI, we have to run the command below in our terminal:

        streamlit run app.py

## Summary of Findings

To test the model both locally and in production to see how the model fares, different genome sequence files were used from the _consensus_sequences_ folder but for the purpose of this summary, we will be using the _consensus_sequence14.fa_ file. Below is a table that captures the results:

<div align = "center">
<table>
        <tr>
                <th>Drugs</th>
                <th>Resistance Result</th>
        </tr>
        <tr>
                <td>Chloroquine</td>
                <td>Sensitive</td>
        </tr>
        <tr>
                <td>Dihydroartemisinin</td>
                <td>Resistant</td>
        </tr>  
        <tr>
                <td>Lumefantrine</td>
                <td>Sensitive</td>
        </tr>
        <tr>
                <td>Quinine</td>
                <td>Sensitive</td>
        </tr>
        <tr>
                <td>Halofantrine</td>
                <td>Resistant</td>
        </tr>
        <tr>
                <td>Piperaquine</td>
                <td>Sensitive</td>
        </tr>

</table>
</div>

## Conclusion

In the future, we hope to achieve the creation of a pipeline where models are automatically built as new drugs are added for testing against different genome sequence.
