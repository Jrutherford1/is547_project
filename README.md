# is547_project

DISCLAIMER: All code and content in this project represent a collaborative effort with large language models (LLMs), including but not limited to OpenAI's ChatGPT, Anthropic's Claude, and xAI's Grok. These tools have been utilized for tasks such as code generation, debugging, drafting, and ideation, with human oversight and modification applied throughout the process.  All contributions were subject to human evaluation, adaptation, and final approval.

Data to be used for the project is available at: https://uofi.app.box.com/folder/212652485445?s=pww3vn4y8plg5jab2fwxw9aulw7csuy2

Git Repository where all code is stored: https://github.com/Jrutherford1/is547_project

# Project Overview
This project contains a workflow and documentation for managing around 2,200 digital documents originally from a WordPress site migration. The main goal is to enhance access and maintain institutional memory by applying consistent naming conventions, improving metadata, adding fixity, and tracking provenance. 

# Project Structure
- Jupyter Notebook: Contains the code used for data processing and analysis.
- /data: Contains the original and processed data files and csv files containing metadata.
- /data_pipline: Contains the code used for data processing and analysis.
- Other documentation: Contains the project requirements and other documents, including this README file and requirements.

# Workflow
1. **Data Ingestion**: 
   - Collect all digital documents from the WordPress site migration.
   - Store them in the `/data/Committees` directory.

2. **Data Processing**:
   - Use the Jupyter Notebook to read and use functions in "data_explore.py" to inspect the data.
   - Ensure directory, copy and list files with data_clean.py.
   - Apply consistent naming conventions to the files by processing with functions from "file_naming.py".  This generates "names.csv".  Then manually inspect the file names using OpenRefine and create new "manually_updated_committee_names.csv" where manual intervention is needed.
   - Finish processing final file names with function from "file_naming.py" and produce "final_updated_committee_names.csv".
   - Use "rename_processed_files function" to rename the files in the `/data/Processed_Committees` directory.
   - Enhance metadata and add fixity checks for each document with "enhance_metadata.py" .


3. **Data Output**:
   - Store processed documents in the `/data/Processed_Commitees` directory.
   - Save metadata in separate json files for easy access and analysis.

# Data Governance
- Committee data are publicly accessible per library policy, generally, however the full set of documentation is behind a login wall here: https://uofi.app.box.com/folder/212652485445?s=pww3vn4y8plg5jab2fwxw9aulw7csuy2
- All data handling, processing, and analysis must comply with institutional data governance policies as outlined in the Library's policy. Note that all committees receive their charge from the Executive Committee, and therefore, the policies for all other committees fall under this umbrella policy: https://www.library.illinois.edu/staff/policies/confidentiality-of-reports-submitted-to-the-library-executive-committee/
