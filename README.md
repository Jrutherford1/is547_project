# is547_project



DISCLAIMER: All code and content in this project represent a collaborative effort with large language models (LLMs), including but not limited to OpenAI's ChatGPT, Anthropic's Claude, and xAI's Grok. These tools have been utilized for tasks such as code generation, debugging, drafting, and ideation, with human oversight and modification applied throughout the process.  All contributions were subject to human evaluation, adaptation, and final approval.

Data to be used for the project is available at: https://uofi.app.box.com/folder/212652485445?s=pww3vn4y8plg5jab2fwxw9aulw7csuy2

Git Repository where all code is stored: https://github.com/Jrutherford1/is547_project

# Project Overview
This project contains a workflow and documentation for managing around 2,200 digital documents originally from a WordPress site migration. The main goal is to enhance access and maintain institutional memory by applying consistent naming conventions, improving metadata, adding fixity, and tracking provenance. This project will allow for additional committee data to be added in the future, should we at some point want to archive another set.  Also, it will work as long as the next set has similar naming conventions and folder structure.

# Project Structure
- Jupyter Notebook: Contains the code used for data processing and analysis.
- /data: Contains the original and processed data files and csv files containing metadata.
- /data_pipline: Contains the code used for data processing and analysis.
- **knowledge_graph_explorer.html**: Interactive visualization of people-document relationships extracted from committee records.
- Other documentation: Contains the project requirements and other documents, including this README file and requirements.

# Knowledge Graph Explorer

An interactive visualization tool that maps relationships between people and documents in the committee archive. Built using NLP extraction and network visualization.

**Quick Start**: Open `knowledge_graph_explorer.html` in your browser to explore the data.

**Documentation**:
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - How to use the visualization (start here!)
- **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** - Tips for presenting to stakeholders
- **[VISUALIZATION_IMPROVEMENTS.md](VISUALIZATION_IMPROVEMENTS.md)** - Technical details of improvements made
- **[NODE_SPACING_FIX.md](NODE_SPACING_FIX.md)** - How the layout algorithm works

**Features**:
- 🔍 Search for specific people or documents
- 🎯 Click any person to see their document connections
- 📊 Real-time statistics and filtering
- 🎨 Clean, professional visualization
- 💾 No server required - runs entirely in browser

# Workflow
1. **Data Ingestion**: 
   - Download all digital documents from the WordPress site migration.
   - IMPORTANT: Manually inspect and make sure each committee folder has Agenda, Minutes, and Related Documents folders and sort the files into these folders. Administrative Counsel and Collection Development Committee were not correctly structured, all others were fine.  
   - Store them in the `/data/Committees` directory.

2. **Data Processing**:
   - Use the Jupyter Notebook to read and use functions in "data_explore.py" to inspect the data.
   - Ensure directory, copy and list files with data_clean.py.
   - Apply consistent naming conventions to the files by processing with functions from "file_naming.py".  This generates "names.csv".  
   - Manually inspect the file names using OpenRefine and create new "manually_updated_committee_names.csv".
   - Finish processing final file names with "build_final_filenames" function from "file_naming.py" and produce "final_updated_committee_names.csv".
   - Use "rename_processed_files function" to rename the files in the `/data/Processed_Committees` directory.
   - Enhance metadata and add fixity checks for each document with "enhance_metadata.py" .


3. **Data Output**:
   - Store processed documents in the `/data/Processed_Commitees` directory.
   - Save metadata in separate json files for easy access and analysis.
   - Apply fixity checks to ensure data integrity.

# Data Governance
- Committee data are publicly accessible per library policy, generally, however the full set of documentation is behind a login wall here: https://uofi.app.box.com/folder/212652485445?s=pww3vn4y8plg5jab2fwxw9aulw7csuy2
- All data handling, processing, and analysis must comply with institutional data governance policies as outlined in the Library's policy. Note that all committees receive their charge from the Executive Committee, and therefore, the policies for all other committees fall under this umbrella policy: https://www.library.illinois.edu/staff/policies/confidentiality-of-reports-submitted-to-the-library-executive-committee/

