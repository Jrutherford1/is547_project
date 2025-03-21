# is547_project

Provenance of the data used in this project:

- The data used in this project is from:

Workflow:
- The data is first downloaded from the source.
- Then I explore the data to understand its structure and contents in data_explore.py
  - Data_explore.py contains several functions to look at the naming, file structure, counts.
- Next I will extract the committee name, file type (Agenda, Minutes, Related Documents) from the data and the file name associated and put this into csv file.
- Then I will process the file names to extract the date.
- At this point I should build a file to hold these items as curated metadata in case I want to do metadata enhancement.
- Then I will rebuild the file names to be consistent like this: committee_name_type_date.ext
- Possibly: I will add fixity to the files???
- I will need to either create/emulate old directory structure???  Or put all into one folder to search??
-------
other notes from beginning of project:

I did exploration of the data noting that the naming conventions are a mess and need a proper scheme, capitalization, etc.  Note the date format at the end, some begin 0 some have no hyphen between last word and beginning of date, some are just by date: 

Questions: do I even want to preserve the folder structure?  YES Would a list of files appropriately named be fine?  YES How to search them? MANUAL IS FINE
There is probably no point in manipulating directory structure (deleting empty folders) if we go about renaming all the files based on that structure. KEEP DIRECTORY STRUCTURE, EVEN EMPTY ONES AS PROVENANCE THAT NOTHING EXISTED IN THOSE CATEGORIES
Metadata enhancement: NOT WORTH THE TIME USING BOX, it can be done for each file type, Excel, PPT, Doc, but each type has its own script, Box does not work with this kind of enhancement, only its proprietary tools.  
Fixity: While Box’s versioning history adds a layer of provenance to the document, and is quite enough for this data, I may do fixity as something to “set” the data as it is and any further changes can be documented and isolated if there is a dispute.

------------------------
3/21/2025 
I have the data in a folder and I have explored the data. I have extracted the committee name, file type, and file name. I have also extracted the date from the file name. I have put this information into a csv file. I have also created a new file name that is consistent with the committee name, file type, and date. I have not added fixity to the files yet. I have not created the old directory structure yet. I have not put all the files into one folder yet. I have not done metadata enhancement. I have not built a file to hold the curated metadata. What I need to do before all that - 
- I need to manually go through the data, now that I have a visual of the data and I need to sort name and type the Administrative council files.  The collection development committee needs sorted by type, Anything "Related Document" needs skipped as no dates exist and are not important for those files, what is important is that they remain with the set of data they are associateed to (by committee), must keep original name.
      - first go back to original data and sort
      - then run the processes again
      - then manually add dates to pertinant files and save a new csv "manually_cleanded_names.csv"   
 